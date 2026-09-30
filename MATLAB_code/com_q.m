%% 六个关节的位置对比：实际位置与期望位置
% 生成两张图：位置误差（6行1列）、位置对比（3行2列）。
% 输入位置单位为 rad，不滤波、不平移位置、不自动修正跟踪延迟。

%% 1. 文件路径和绘图设置
script_dir = fileparts(mfilename('fullpath'));
root_dir = fileparts(script_dir);
actual_file = fullfile(root_dir, 'data_out', 'circle_R400_SR4_100Hz_25.txt');
reference_file = fullfile(root_dir, 'data_in', 'circle_R400_joint_poses_SR4_V50.txt');

% 参考轨迹按用户指定的1000 Hz执行，每行间隔1 ms。
% 不使用输入文件的时间列，直接用采样序号生成时间轴。
reference_hz = 1000;
angle_unit = 'deg';          % 'deg'：角度；'rad'：弧度。
show_figures = usejava('desktop'); % MATLAB桌面运行时显示窗口，批处理只保存图片。

output_dir = fullfile(root_dir, 'data_out', ...
    'circle_R400_SR4_100Hz_25_comparison_index');

%% 2. 读取实际位置
% 实际文件是CSV，pos列形如 "[q1,q2,q3,q4,q5,q6]"。
actual_options = detectImportOptions(actual_file, 'Delimiter', ',');
if ~all(ismember({'index', 'pos'}, actual_options.VariableNames))
    error('实际日志必须包含 index、pos 两列。');
end
actual_options.SelectedVariableNames = {'index', 'pos'};
actual_options = setvartype(actual_options, 'pos', 'string');
actual_data = readtable(actual_file, actual_options);

sample_index = actual_data.index;            % C++日志从0开始编号。
actual_q = zeros(height(actual_data), 6);
position_text = string(actual_data.pos);
for i = 1:height(actual_data)
    % 去掉方括号，再按逗号分隔读取六个数，不使用eval。
    text = erase(position_text(i), ["[", "]"]);
    values = sscanf(char(text), '%f,');
    if numel(values) ~= 6 || any(~isfinite(values))
        error('实际日志第 %d 个数据行的pos不是六个有效数值。', i);
    end
    actual_q(i, :) = values.';
end
if height(actual_data) < 2
    error('实际日志需要至少两行数据。');
end

%% 3. 读取原始参考位置
% 只读取第2~7列（q1~q6），跳过第一列时间和后面的速度列。
reference_options = detectImportOptions(reference_file, 'FileType', 'text', 'Delimiter', '\t');
if numel(reference_options.VariableNames) < 7
    error('参考文件至少需要七列，第2~7列为关节位置。');
end
reference_options.SelectedVariableNames = reference_options.VariableNames(2:7);
reference_q = readmatrix(reference_file, reference_options);
if ~isfinite(reference_hz) || reference_hz <= 0
    error('reference_hz必须是有限的正数。');
end
if size(reference_q, 1) < 2 || size(reference_q, 2) ~= 6 || any(~isfinite(reference_q(:)))
    error('参考位置需要至少两行、六列，数值必须全部有限。');
end
reference_time = (0:size(reference_q, 1)-1).' / reference_hz;

fprintf('实际记录：%d个数据点。\n', size(actual_q, 1));
fprintf('参考轨迹：%d个数据点，按 %g Hz执行，结束时间 %.6f s，间隔 %.6f s。\n', ...
    size(reference_q, 1), reference_hz, reference_time(end), 1 / reference_hz);

%% 4. 对齐期望位置和实际位置
if any(~isfinite(sample_index)) || any(sample_index < 0) || ...
        any(sample_index ~= floor(sample_index)) || any(diff(sample_index) <= 0)
    error('index必须是从0起的非负整数，并严格递增。');
end
% 日志index=0对应参考第1行；按相同采样序号直接相减，不插值。
valid = sample_index < size(reference_q, 1);
reference_rows = sample_index(valid) + 1;
reference_at_actual = reference_q(reference_rows, :);
plot_time = sample_index(valid) / reference_hz;
time_label = '时间 (s)';
alignment_label = sprintf('按采样序号对齐，采样率 %g Hz', reference_hz);
if nnz(valid) < 2
    error('两份数据没有足够的可对齐采样点。');
end
if any(~valid)
    fprintf('有 %d 个实际采样点超出参考范围，未参与误差计算。\n', nnz(~valid));
end
actual_q = actual_q(valid, :);

% 两张图统一使用用户选择的单位。
if strcmp(angle_unit, 'deg')
    scale = 180 / pi;
elseif strcmp(angle_unit, 'rad')
    scale = 1;
else
    error('angle_unit只能设置为deg或rad。');
end
actual_q = actual_q * scale;
reference_q = reference_q * scale;
reference_at_actual = reference_at_actual * scale;
position_error = actual_q - reference_at_actual; % 正值：实际位置大于期望位置。

%% 5. 位置误差：一张图，6行1列
visibility = 'off';
if show_figures
    visibility = 'on';
end
error_figure = figure('Name', '六关节位置误差', 'Color', 'w', ...
    'Position', [80, 50, 1000, 1100], 'Visible', visibility);
error_axes = gobjects(6, 1);
for joint = 1:6
    error_axes(joint) = subplot(6, 1, joint);
    plot(plot_time, position_error(:, joint), 'b-', 'LineWidth', 1.0);
    hold on;
    yline(0, 'k:', 'LineWidth', 0.8);
    grid on;
    ylabel(sprintf('关节%d (%s)', joint, angle_unit));
    xlim([plot_time(1), plot_time(end)]);
    set(gca, 'FontSize', 10);
end
xlabel(error_axes(6), time_label);
linkaxes(error_axes, 'x');
sgtitle({'位置误差 = 实际位置 - 期望位置', alignment_label}, 'FontSize', 14);

%% 6. 原始期望位置与实际位置：一张图，3行2列
comparison_figure = figure('Name', '六关节位置对比', 'Color', 'w', ...
    'Position', [120, 80, 1300, 850], 'Visible', visibility);
comparison_axes = gobjects(6, 1);
for joint = 1:6
    comparison_axes(joint) = subplot(3, 2, joint);
    % 期望曲线直接使用原始参考采样点，不使用插值后的数组绘图。
    plot(reference_time, reference_q(:, joint), 'r--', 'LineWidth', 1.5);
    hold on;
    plot(plot_time, actual_q(:, joint), 'b-', 'LineWidth', 1.0);
    grid on;
    title(sprintf('关节 %d', joint));
    xlabel(time_label);
    ylabel(['位置 (' angle_unit ')']);
    legend('原始期望位置', '实际位置', 'Location', 'best');
    xlim([reference_time(1), reference_time(end)]);
    set(gca, 'FontSize', 10);
end
linkaxes(comparison_axes, 'x');
sgtitle({'原始期望位置与实际位置对比', alignment_label}, 'FontSize', 14);

%% 7. 保存图片和MATLAB可编辑图形，显示各关节误差统计
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end
drawnow;
exportgraphics(error_figure, fullfile(output_dir, 'position_error_6x1.png'), 'Resolution', 200);
exportgraphics(comparison_figure, fullfile(output_dir, 'position_comparison_3x2.png'), 'Resolution', 200);
savefig(error_figure, fullfile(output_dir, 'position_error_6x1.fig'));
savefig(comparison_figure, fullfile(output_dir, 'position_comparison_3x2.fig'));

rmse = sqrt(mean(position_error.^2, 1));
max_error = max(abs(position_error), [], 1);
fprintf('\n位置误差统计，单位：%s\n', angle_unit);
disp(table((1:6).', rmse.', max_error.', ...
    'VariableNames', {'Joint', 'RMSE', 'MaxAbsError'}));
fprintf('图像保存到：%s\n', output_dir);
