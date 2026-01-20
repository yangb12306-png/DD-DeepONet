
%% 生成数据
% % 设置随机种子（保证每次运行结果一致）
% rng(43, 'twister'); 
% 
% % 生成随机数据
% num_samples = 10100; % 样本数量
% 
% % 生成 dx1，范围 [4, 8]，保留两位小数
% dx1 = round(4 + (8 - 4) * rand(num_samples, 1), 2);
% 
% % 生成 dy1，范围 [1, 3]，保留两位小数
% dy1 = round(1 + (3 - 1) * rand(num_samples, 1), 2);
% 
% % 生成 dx2，范围 [1, 3]，保留两位小数
% dx2 = round(1 + (3 - 1) * rand(num_samples, 1), 2);
% 
% % 生成 dy2，范围 [6, 10]，保留两位小数
% dy2 = round(6 + (10 - 6) * rand(num_samples, 1), 2);
% 
% % 生成 dy3，范围 [1, 3]，保留两位小数
% dy3 = round(1 + (3 - 1) * rand(num_samples, 1), 2);
% 
% % 生成 dx3，范围 [6, 10]，保留两位小数
% dx3 = round(6 + (10 - 6) * rand(num_samples, 1), 2);
% 
% % 生成 U0，范围 [0.1, 0.35]，保留两位小数
% U0 = round(0.1 + (0.35 - 0.1) * rand(num_samples, 1), 2);
% 
% % 生成 U1，范围 [0.01, 0.35]，保留两位小数
% U1 = round(0.01 + (0.35 - 0.1) * rand(num_samples, 1), 2);
% U1 = min(U1, U0);
% 
% % % 生成 U_in，从指定集合中随机选择
% % U_in_values = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1];
% % U_in = U_in_values(randi(length(U_in_values), num_samples, 1));
% % U_in = 0.1 * ones(num_samples);
% 
% % 将所有变量拼接为一个矩阵
% data = [dx1, dy1, dx2, dy2, dx3, dy3, U0, U1];
% 
% % 打开文件
% fileID = fopen('data.txt', 'w');
% 
% % 写入数据
% for i = 1:num_samples
%     fprintf(fileID, '%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f\n', dx1(i), dy1(i), dx2(i), dy2(i), dx3(i), dy3(i), U0(i), U1(i));
% end
% 
% % 关闭文件
% fclose(fileID);

%% 运行代码
% clc
% clear
% 
% start_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
% non_convergence = [];
% data_Pipe = load('data_skewing_10100.txt');
% 
% for data_id = 1:10100
%     try
%         data = data_Pipe(data_id, :);
%         dx1 = data(1); dy1 = data(2); dx2 = data(3); dy2 = data(4); dx3 = data(5); dy3 = data(6); U0 = data(7); U1 = data(8);
%         model = Pipe_creat(dx1, dy1, dx2, dy2, dx3, dy3, U0, U1, data_id);
%         disp(['成功处理 data_id: ', num2str(data_id)]);
%         clear model;
%     catch ME  % 捕获异常
%         % 检查错误类型是否为不收敛
%         if contains(ME.message, '返回的解不收敛') || contains(ME.message, '找不到解')
%             non_convergence = [non_convergence; data_id];  % 记录不收敛的 data_id
%             disp(['data_id ', num2str(data_id), ' 不收敛，已跳过']);
%         else
%             % 其他错误（如几何错误、内存不足）直接报错
%             rethrow(ME);
%         end
%         
%         % 确保清除残留模型
%         if exist('model', 'var'), clear model; end
%     end
% end
% 
% % 保存不收敛样本的 data_id
% save('non_convergence_skewing.mat', 'non_convergence');
% 
% end_time = datetime('now', 'Format', 'yyyy-MM-dd HH:mm:ss.SSS');
% 
% elapsed_time = seconds(end_time - start_time);
% fprintf('代码运行时间: %.6f 秒\n', elapsed_time);

% U_0.1
% 不收敛的 21, 838, 1426, 2099, 2681, 2740, 2984, 3163 
% 代码运行时间：16951.631000 秒

% 10100:
% 运行代码时间：55692.731000 秒

% 50000:
% 代码运行时间: 261104.159000 秒

% 10100: skewing (68个不收敛)
% 代码运行时间: 55350.277000 秒


%% 将data.txt中不收敛的行去掉
clc
clear
data = load('data_skewing_10100.txt');
non_convergence = load('non_convergence_skewing.mat');
row_indices = non_convergence.non_convergence;

% 确保row_indices是列向量并排序（从大到小删除，避免索引错位）
row_indices = unique(row_indices(:));
row_indices = sort(row_indices, 'descend');  % 从大到小排序

fprintf('找到需要移除的行数: %d\n', length(row_indices));

% 移除行
filtered_data = data;
filtered_data(row_indices, :) = [];  % 删除指定行

% 4. 显示结果
fprintf('移除后的数据大小: %d × %d\n', size(filtered_data));
fprintf('共移除了 %d 行数据\n', size(data, 1) - size(filtered_data, 1));

% 5. 可选：保存处理后的数据
writematrix(filtered_data, 'data_skewing_10032.txt', 'Delimiter', 'tab');




