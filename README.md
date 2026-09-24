# 机器人NMPC控制器实现
NMPC的非线性值得是预测模型始终为：

$$
x_{i+1}=F(x_i,u_i)
$$

之前的线性MPC代码，每个控制周期只在线性化点$(x_k,u_{k-1})$计算一次$A_k、B_k$，然后整个预测域使用同一组矩阵$ x_{i+1}=A_kx_i+B_ku_i $进行计算，因此这个代码相当于将系统永久替换为一个线性模型，也就是说凡此优化内部仍然是一个冻结的线性模型    

对于非线性MPC控制，依然定义机器人状态为：  

$$

x=\begin{bmatrix}
q \\
\dot{q}
\end{bmatrix}

$$

因此有：

$$

\dot{x}=\begin{bmatrix}
\dot{q} \\
\ddot{q}
\end{bmatrix}

$$

非线性模型用欧拉Euler离散展开为：  

$$

x_{i+1}=F(x_i,u_i)=\begin{bmatrix}
 q_{i+1}\\
\dot{q}_{i+1}
\end{bmatrix}=\begin{bmatrix}
q_{k}+T_s\dot{q}_k \\
\dot{q}_k+T_sABA(q_k,\dot{q}_k,\tau_k)
\end{bmatrix}

$$

这样就可以轻易得到**下一步的状态**

控制器的输出为关节力矩，也就是控制输入

设计代价函数： 

在每一个预测时刻展开，需要满足代价函数：  

$$

J=\sum_{k=0}^{N-1}[(q_k-q_k^{ref})^TQ(q_k-q_k^{ref})+(\dot{q}_k-\dot{q}_k^{ref})^TQ(\dot{q}_k-\dot{q}_k^{ref})+\tau_k^TR\tau_k] 

$$

这个代价函数其实跟下面的代价函数是一样的，只不过下面的代价函数是在整个预测区间进行展开计算：

$$

J=X_k^T\bar{Q}X_k+U_k^T\bar{R}U_k

$$

这两个代价函数就是说希望机器人后面的每一步都尽量靠近期望轨迹，同时不希望控制力矩太大  

## LTV-MPC

一个控制周期内的步骤：  
1、读取当前机器人状态，组成当前时刻的状态变量：

$$
x_0=x_{means}=\begin{bmatrix}
 q\\
\dot{q}
\end{bmatrix}
$$

2、使用逆动力学RNEA获得参考力矩，因为还没求出来最新的最优力矩，因此需要先有一串名义控制力矩

$$
\bar{U}=\begin{bmatrix}
\bar{u}_0 \\
 \bar{u}_1\\
... \\
\bar{u}_{N-1}
\end{bmatrix}
$$

可以使用上一周期的MPC结果向前移动一位，也就是说，第一个周期用RNEA获得参考力矩，后面每个周期将结果向前移动一位  

假设预测长度为20,那么第一个控制周期就要从输入的轨迹文件中提取20个参考状态，然后用每个参考状态计算RNEA，得到这个矩阵，如果最后只剩下一个状态，那就将这个状态循环20次  

3、生成名义预测轨迹  
根据上面两步可以得到 $q、\dot{q}、\tau$ ，因此这里就可以计算出来20组 $\ddot{q}$ ，然后通过非线性模型用欧拉Euler离散展开就可以得到下一步的状态，也就可以得到如果按照目前的 $\tau$ 进行运动，机器人未来的大致运动轨迹  

这一步得到的20个状态X就是名义预测轨迹  

4、上一次的ABA计算的结果是：如果按照当前猜测，未来会走到哪里，这一次需要重新计算ABA，进行前向差分，得到如果稍微改变状态或者控制力矩，未来会如何改变，也就是说，MPC需要获得如何修改控制力矩，才能使得控制效果最好的信息  
也就是分别对每一次的结果算产生一点偏差，后续结果会如何变化，也就是得到A和B矩阵，这样MPC就可以知道改动力矩，下一步十二维的状态变量会发生什么，改动十二维状态变量x，下一步六维力矩会发生什么变化  

同时这一步也是后面QP优化器能够正常工作的原因，因为QP优化器求解出来的是力矩，

假设预测域为20，因此需要求解20个A_i和20个B_i，对于一个预测点，一个Ai需要扰动12次（Ai为12列），一个Bi需要扰动6次，再加上上一次的ABA计算结果，一共是19次ABA计算，因此20个预测域一共会进行20*19=380次ABA计算

5、将未来所有状态写为未来控制量的函数，加入输入的参考轨迹为：$q_1^{ref},q_2^{ref},q_3^{ref}...$ 与速度  $\dot{q}_1^{ref},\dot{q}_2^{ref},\dot{q}_3^{ref}...$，就可以组成期望状态 $x_i^{ref}$,变成大型矩阵，为：  

$$
X^{ref}=\begin{bmatrix}
 x_1^{ref}\\
  x_2^{ref}\\
... \\
 x_N^{ref}
\end{bmatrix}
$$

此时名义预测状态为：  

$$
\bar{X}=\begin{bmatrix}
 \bar x_1\\
 \bar x_2\\
... \\
 \bar x_N
\end{bmatrix}
$$

因此，真正的预测状态为：$X = \bar X+\Delta  X$，即 $X=\bar X +\Phi \delta x_0+\Gamma \Delta U$,注意此时的U再下一步可以算出

6、构造代价函数，求解输出为 $\Delta U$ ，并且可以将代价函数转换为QP问题求解

# 部分代码解释

## A与B的矩阵计算

代码来自： `Robot_NMPC/src/pinocchio_fun.cpp` ，函数 `com_Mat_A_B`

控制器的状态维度为12维（输入位置和输入速度），控制维度为6维（输出力矩）

在这个代码的计算ABA函数的代码中，根据非线性机器人动力学可以得到： $ x_{k+1}=f(x_k,u_k) $ ，下一步就需要线性化得到 $ x_{k+1}=Ax_k+Bu_k $，也就是在当前轨迹附近进行局部线性化，即： $ \delta x_{k+1}=A\delta x_{k}+B\delta u_{k} $ ，这里有：

$$
\begin{matrix}A=\frac{\delta f}{\delta x} 
 \\
B=\frac{\delta f}{\delta u} 
\end{matrix}
$$

使用前向有限差分来进行计算，在每一个预测点都要计算这个矩阵，也就是Linear Time-Varying model，LTV 模型

### 什么是有限差分
对于一个简单函数：$ y=f(x) $，理论导数为：

$$

f'(x)=\lim_{h \to 0} \frac{f(x+h)-f(x)}{h} 

$$

在计算机里面，将 $ h=10^{-6} $ ,就可以得到前向有限差分的计算公式：

$$

f'(x)\approx \frac{f(x+h)-f(x)}{h} 

$$

## 预测矩阵组装
该部分主要在代码文件： `src\MPC_Matrices.cpp`

在前面已经得到了每个预测点附近的局部线性模型，下面就要将未来第1步、第2步...到第N步的状态全部写为“当前状态 $\Delta{x_0}$ ”和“未来所有控制量 $ \Delta{U} $ ”的函数，即

$$
\Delta X = \Phi \Delta x_0  +\Gamma \Delta U
$$

未来所有修正控制量为：

$$
\Delta U=\begin{bmatrix}\Delta u_0
\\
 \Delta u_1\\
 ...\\
\Delta u_{19}
\end{bmatrix}
$$

未来所有的状态组合矩阵为：

$$
\Delta X=\begin{bmatrix}\Delta x_1
\\
 \Delta x_2\\
 ...\\
\Delta x_{20}
\end{bmatrix}
$$

已知结果 $ \Delta X $ 为240 * 1的矩阵，输入 $x_0$ 为12 * 1的矩阵，$\Delta U$ 为120 * 1的矩阵，可以得到：

$$
\left\{\begin{matrix}
 \Phi \in \mathbb{R}^{240\times 12} \\
\Gamma \in \mathbb{R}^{240\times 120}
\end{matrix}\right.
$$

其中，$ \Phi $ 用来描述当前状态误差如何传播到未来； $\Gamma$ 用来描述未来控制量如何影响未来状态

下面解释代码：

```C++
for(int i = 0; i < N; i++){
    //delta_x_i = A_i*delta_x_{i}+B_i*delta_u_i
    Phi_i = Mat_A[i] * Phi_i;
    Gamma_i = Mat_A[i] * Gamma_i;
    Gamma_i.block(0, i*p, n, p) += Mat_B[i];
    
    Phi.block(i*n, 0, n, n) = Phi_i;
    Gamma.block(i*n, 0, n, N*p) = Gamma_i;
}
```

第一步的计算公式：

$$
\Delta x_1 = A_0\Delta x_0 +  B_0\Delta u_0
$$

第二步的计算公式：

$$
\Delta x_2 = A_1\Delta x_1 +  B_1\Delta u_1
$$

代入第一步的计算公式，可以得到：

$$
\Delta x_2 = A_1 ( A_0\Delta x_0 +  B_0\Delta u_0 )+  B_1\Delta u_1
$$

展开可以得到：

$$
\Delta x_2 = A_1 A_0\Delta x_0 + A_1 B_0\Delta u_0 +  B_1\Delta u_1
$$

第三步的计算公式：

$$
\Delta x_3 = A_2\Delta x_2 +  B_2\Delta u_2
$$

继续代入，可以得到：

$$
\Delta x_3 = A_2 A_1 A_0\Delta x_0 +A_2 A_1 B_0\Delta u_0 + A_2 B_1\Delta u_1 + B_2 \Delta u_2
$$

最终可得：

$$
\Delta X=\begin{bmatrix}
 A_0\\
A_1 A_0 \\
A_2 A_1 A_0
\end{bmatrix}\Delta x_0+\begin{bmatrix}
 B_0 &0  & 0\\
A_1B_0  & B_1 &0 \\
A_2A_1B_0  & A_2B_1 &B_2
\end{bmatrix}\Delta U
$$

首先创建大矩阵Phi，之后Phi_i就变为：

$$
\Phi_i=A_0I=A_0
$$

第二次循环变为：

$$
\Phi_i=A_1A_0
$$

第三次循环变为：

$$
\Phi_i=A_2A_1A_0
$$

对于gamma，第一次循环，当i=0时：

$$
\Phi_i=A_0I=A_0
$$

$$
\Gamma_i = A_0\times0 = 0
$$

$$
\Gamma_i = \begin{bmatrix}
 B_0 & 0 &0
\end{bmatrix}
$$

第二次循环，有：

$$
\Gamma_i = A_1B_0
$$

$$
\Gamma_i = \begin{bmatrix}
  A_1B_0 & 0 &0
\end{bmatrix}
$$

然后下一步， ` Gamma_i.block(0, i*p, n, p) += Mat_B[i];` ，使得上面的矩阵进一步变为：

$$
\Gamma_i = \begin{bmatrix}
  A_1B_0 & B_1 &0
\end{bmatrix}
$$

第三次循环同理

随后，就是建立矩阵Q和终端矩阵F，其中Q为对角矩阵，前6个数字对应关节位置误差，后6个数字对应关节速度误差，数字越大说明对这个地方的误差越在意

F矩阵为终端权重，是希望如果中间可以有一定误差，但是预测区间末端希望机器人尽可能接近目标，因此可以选择F>Q

R矩阵用来惩罚控制量，如果R很小，则MPC为了减少轨迹误差，可以非常激进的改变控制力矩，因此R更看重控制平滑、控制的代价、能量损耗

## 代价函数
MPC的代价函数为：

$$
J=\sum_{k=1}^{N-1} \Delta x_k^TQ\Delta x_k+\Delta x_N^TF\Delta x_N+\sum_{k=1}^{N-1}\Delta u_k^TR\Delta u_k
$$

可以得到：

$$
J=\Delta X^TQ_{bar}\Delta X+\Delta U^TR_{bar}\Delta U
$$

已知：

$$
\Delta X = \Phi \Delta x_0  +\Gamma \Delta U
$$

所以可以得到：

$$
J=( \Phi \Delta x_0  +\Gamma \Delta U)^TQ_{bar}( \Phi \Delta x_0  +\Gamma \Delta U)+\Delta U^TR_{bar}\Delta U
$$

继续展开，可以得到：

$$
J = \Delta U^T(\Gamma^TQ_{bar}\Gamma+R_{bar})\Delta U+2\Delta x_0^T\Phi^TQ_{bar}\Gamma\Delta U
$$

这里可以构造QP规划，为：

$$
\min_{\Delta U}\frac{1}{2}  \Delta U^TH\Delta U+g^T\Delta U
$$

可以构造：

$$
H = 2(\Gamma^TQ_{bar}\Gamma+R_{bar})
$$

$$
g = 2\Gamma^TQ_{bar}\Phi\Delta x_0
$$

这时就可以写函数Prediction，即代码文件 `src\Prediction.cpp`

## 控制器代码
代码来自 `src\NMPC_control.cpp`  

该代码主要是输入机器人当前轨迹和状态，输出当前应该发送的六维关节力矩

将所有模块进行整合，控制器总体步骤为：非线性预测、轨迹线性化、QP优化、只执行第一步，也就是再每个控制周期只在线性化轨迹附近求一次QP修正，而不是在一个控制周期内部不断收敛直到完全收敛

首先就是构造函数，这里用到了参数 `nominal_tau`，为当前这一轮优化围绕的名义控制轨迹，也就是说，QP并不是直接从0开始求解U，而是从已有的一个控制序列U进行较小的修正

```C++
void ROKAE_NMPC::initial_nom_ctrl(
    const std::vector<Eigen::VectorXd>& control_ref
)
{
    for(int i = 0; i < N; ++i) {
        nominal_tau[i] = control_ref[i];
    }
}
```

因为第一个控制周期没有上一轮的最优解，因此第一个控制周期使用目标状态的RNEA力矩初始化U_bar,也就是预测区域内部20个参考状态通过RNEA计算得到的参考力矩

## 机器人零力矩Mujoco仿真

首先需要在xml代码中，将重力项改为0，具体代码为 `xml\ROKAE_SR4.XML` 的 ` <option gravity="0 0 0" timestep="0.02"/>` 正常为-9.81，这里改为0

详细零力矩仿真的代码可以查看 `python_code\zero_torque_sim.py`  可以查看具体注释了解详细内容

下面附上仿真图

<div align="center">
    <img src="fig\zero_tau.gif">
    <br>
    零力矩输入仿真（重力补偿+摩擦补偿）
</div>

## 通过pybind进行Mujoco仿真

因为Mujoco的原生simulate环境不支持C++，因此用pybind生成.so文件作为动态链接库，放到python环境下进行仿真执行，具体链接文件查看 `src\python_bindings.cpp`

也就是说，以后需要在python环境下调用C++的控制器，都可以使用这种方法

pybind的基本语法为： `PYBIND11_MODULE(test, m)` 表示创建一个Python模块，名字为test

`m.def("add",&add)` 表示将C++的add()函数注册给Python，在Python中也叫add

之后再python代码中 `import test` 就可以动态加载.so文件

```c++
class MPCController {
    pinocchioFun dynamics;
    ROKAE_NMPC controller;
```

因为NMPC与动力学是绑定在一起的，在创建NMPC控制器的时候，需要pinocchioFun建立动力学，因此需要创建一个class

```c++
explicit MPCController(const std::string& urdf_path, double timestep, int horizon)
    : dynamics(urdf_path), controller(dynamics, timestep, horizon) {}
```

这个构造函数，表示在python代码后期加入pybind的动态链接文件时，需要输入

```python
mpc = rokae_mpc.MPCController(
    "SR4.urdf",
    0.001,
    40
)
```
## 什么是名义点 为什么需要名义点

第一次运行控制器的时候，需要名义值作为初始猜测；从第二次开始，直接利用上一次MPC已经算出来的最优解作为名义值

一般来说，需要让新的名义控制序列的最后一个值等于上一轮最优控制序列的最后一个值，整体前移

假设预测3步，上一轮MPC已经算出最优控制序列为： $ U^{*}=[10,11,12] $，但是机器人只执行第一步算出的u=10，下一轮MPC开始之前，选择初始方案为 $ \bar{U} _{new}=[11,12,12] $。这样有了这一组名义控制量，就可以推算后续的名义状态

局部线性化MPC：在某一个状态附近，将机器人动力学近似看为线性的

# 整体流程

设期望状态为x_{ref},当前实际状态为x_{current}  
在第一个时刻t_0，期望轨迹和真实状态同时进入，假设预测长度为N=20，MPC的控制频率为1ms（1000Hz）

假设Ts=1ms，预测长度为20，当前时刻为t_0，因此MPC往未来看的时间点就是t_0，t_0+1ms，t_0+20ms等等

计算下来，这其中一共21个状态，一共需要施加20个力矩  

首先机器人第一个实际状态开始进入，得到：x_0=x_{current}

同时，参考轨迹的20个期望状态进入控制器，因此目前MPC知道两件事，首先是x_0得到机器人当前的位置和状态；20个期望状态告诉MPC希望未来20ms去哪里  
目前MPC需要解决u_1到u_19到底是多少  
需要首先给MPC一组名义力矩，即候选力矩， $ \bar{U}=[\bar{u}_0,\bar{u}_2,\bar{u}_3...\bar{u}_{19}] $  

因为现在已经有了 $\bar{u}_0$ 因此动力学模型就可以根据 $ \bar{x}_0 $ 计算出来下一步的 $\bar{x}_1$ ，以此类推，可以计算出来未来20ms的状态  
之后，比较未来状态和期望状态，得到每一步的误差e  

但是只知道误差还不够，还需要知道将某一个力矩修改一点，未来的轨迹会发生多大的变化，因此就需要在每一个预测节点附近进行线性化，即公式：$ \Delta x_{k+1} = A_k \Delta x_k + B_k \Delta u_k $ ,其中， $ A_k = \frac{\partial F}{\partial x} $ 表示状态更改一点，会产生什么影响； $ B_k = \frac{\partial F}{\partial u} $ 表示力矩改一点，会产生什么影响  

在第一轮计算里面，因为 $\bar{x}_0=x_0=x_{current}$ 因此此时的 $\Delta x_0 = x_0-\bar{x}_0=0 $ ,因此第一轮的MPC主要考虑 $ \Delta{X} = \Gamma \Delta U $ ，但是在实际代码里面，还是需要 $\Delta X = \Phi \Delta x_0 + \Gamma \Delta U$

之后进入代价函数，实际预测状态近似为 $ X \approx \bar{X} + \Delta X $ ，得到误差为： $E = X - X_{\text{ref}}$ 即 $ E = \bar{X} + \Delta X - X_{\text{ref}} $

代价函数=跟踪误差+控制代价，优化器最终会得到 $ \Delta U^* = [\Delta u_0^*, \Delta u_1^*, \ldots] $ ,即 $ U^* = \bar{U} + \Delta U^*  $

但是机器人只执行第一步 $ U_0^* $  

假如控制器频率为100Hz，假如未来N=3，那么第一轮MPC看到的是10ms，20ms和30ms这三个未来时刻  


# 构造函数与普通初始化函数的区别

构造函数能够保证一个对象从刚创建出来的那一刻开始，就是合法、可用且完整的

如果不加入构造函数，就需要一个init函数进行初始化，例如下面构造函数：

```c++
pinocchioFun::pinocchioFun(
    const std::string& urdf_path)
    :model(),data(model)
{
    DOF = 6; // 自由度
    q_step = 1e-6; // 前向差分计算步长
    dq_step = 1e-6;
    tau_step = 1e-4;
    
    pinocchio::urdf::buildModel(urdf_path, model);
    data = pinocchio::Data(model);
}
```

上面的构造函数完全可以写成：

```c++
void pinocchioFun::init(const std::string& urdf_path)
{
    DOF = 6;

    q_step = 1e-6;
    dq_step = 1e-6;
    tau_step = 1e-4;

    pinocchio::urdf::buildModel(urdf_path, model);
    data = pinocchio::Data(model);
}
```

但是构造函数是自动且强制发生的，普通函数不是  
如果使用普通函数，后续使用class的时候，代码应该是：
```c++
pinocchioFun dynamics;

dynamics.init("sr4.urdf")
```

使用构造函数，后续代码是：
```c++
pinocchioFun dynamics("sr4.urdf")

```

# 函数功能详解

文件：

src\pinocchio_fun.cpp：  
函数compute_aba(const Eigen::VectorXd& state,const Eigen::VectorXd& controldouble Ts)  
用来输出下一时刻的状态next_state

函数compute_held_state(const Eigen::VectorXd& state,const Eigen::VectorXd& control, double duration)  
用来保持预测区间的输入，即假如控制频率为100Hz，在这个100Hz（0.01s）的duration内保持相同的控制力矩  
step用来计算这一个控制频率区间里面有多少时间步，按照时间步进行差分计算，最后得到的predicted是0.01s之后的系统状态（state）

