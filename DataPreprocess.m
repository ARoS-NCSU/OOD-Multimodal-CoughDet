clc
clear all

sub_ids = ["005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015", "016", "017"];
trials = ["1", "2","3"];

for i = 1: length(sub_ids)
    sub_id = sub_ids(i);
    fileID = fopen("Murata_IRB_Data/"+sub_id+"/"+"sync_time.txt",'r');
    C = textscan(fileID, '%s', 'Delimiter', ' ');
    for j = 1: length(trials)
        trial = trials(j);
        file_name = "processedData/v2/"+sub_id+"_"+trial+"_multi.mat";
        
        %% Load data
        %IMU
        trial_name = ["Trial_1_No_Talking","Trial_2_Talking","Trial_3_Nonverbal"];
        if trial == "1"
            t1 = readtable("Murata_IRB_Data\"+sub_id+"\"+trial_name(1)+"\Accelerometer.csv");
            t2 = readtable("Murata_IRB_Data\"+sub_id+"\"+trial_name(1)+"\Gyroscope.csv");
            t3 = readtable("Murata_IRB_Data\"+sub_id+"\"+trial_name(1)+"\Magnetometer.csv");
        elseif trial == "2"
            t1 = readtable("Murata_IRB_Data\"+sub_id+"\"+trial_name(2)+"\Accelerometer.csv");
            t2 = readtable("Murata_IRB_Data\"+sub_id+"\"+trial_name(2)+"\Gyroscope.csv");
            t3 = readtable("Murata_IRB_Data\"+sub_id+"\"+trial_name(2)+"\Magnetometer.csv");
        elseif trial == "3"
            t1 = readtable("Murata_IRB_Data\"+sub_id+"\"+trial_name(3)+"\Accelerometer.csv");
            t2 = readtable("Murata_IRB_Data\"+sub_id+"\"+trial_name(3)+"\Gyroscope.csv");
            t3 = readtable("Murata_IRB_Data\"+sub_id+"\"+trial_name(3)+"\Magnetometer.csv");
        end
        
        %Audio & Annotation
        source = load("processedData/v1/"+sub_id+"_"+trial+"_"+"single.mat");
        
        %% Compute time
        data_time = linspace(0,double(length(source.data)/source.samplerate),length(source.data));
        data = source.data;
        label_time = linspace(0,double(length(source.label)/source.labelrate),length(source.label));
        label = source.label;
        
        t1_t = t1.elapsed_s_;
        t1_x = t1.x_axis_g_;
        t2_t = t2.elapsed_s_;
        t2_x = t2.x_axis_deg_s_;
        t3_t = t3.elapsed_s_;
        t3_x = t3.x_axis_T_;
       
        
        %% Sync
        data_start = str2double(C{1,1}(5+8*(str2num(trial)-1)));
        imu_start = str2double(C{1,1}(8+8*(str2num(trial)-1)));
        
%         plot(data_time-data_start,data/max(data));
%         hold on
%         plot(t1_t-imu_start, t1_x)
%         
        %% Save data
        imu_end = data_time(end) - data_start + imu_start;
        
        [t1_start_val,t1_start_idx]=min(abs(t1_t-imu_start));
        [t2_start_val,t2_start_idx]=min(abs(t2_t-imu_start));
        [t3_start_val,t3_start_idx]=min(abs(t3_t-imu_start));
        
        [t1_end_val,t1_end_idx]=min(abs(t1_t-imu_end));
        [t2_end_val,t2_end_idx]=min(abs(t2_t-imu_end));
        [t3_end_val,t3_end_idx]=min(abs(t3_t-imu_end));
        
        acc_time = t1_t(t1_start_idx:t1_end_idx) - imu_start;
        acc_x = t1.x_axis_g_(t1_start_idx:t1_end_idx);
        acc_y = t1.y_axis_g_(t1_start_idx:t1_end_idx);
        acc_z = t1.z_axis_g_(t1_start_idx:t1_end_idx);
        
        gyro_time = t2_t(t2_start_idx:t2_end_idx) - imu_start;
        gyro_x = t2.x_axis_deg_s_(t2_start_idx:t2_end_idx);
        gyro_y = t2.y_axis_deg_s_(t2_start_idx:t2_end_idx);
        gyro_z = t2.z_axis_deg_s_(t2_start_idx:t2_end_idx);
        
        mag_time = t3_t(t3_start_idx:t3_end_idx) - imu_start;
        mag_x = t3.x_axis_T_(t3_start_idx:t3_end_idx);
        mag_y = t3.y_axis_T_(t3_start_idx:t3_end_idx);
        mag_z = t3.z_axis_T_(t3_start_idx:t3_end_idx);
        
        [data_start_val,data_start_idx]=min(abs(data_time-data_start));
        audio_time = data_time(data_start_idx:end) - data_start;
        audio_data = data(data_start_idx:end);
        
        [label_start_val,label_start_idx]=min(abs(label_time-data_start));
        label = label(label_start_idx:end,:);
        
        name = source.labeled_file;
        audio_sample_rate = source.samplerate;
        label_sample_rate = source.labelrate;
        imu_sample_rate = [100,100,25];
        
        
        save(file_name,"acc_time","acc_x","acc_y","acc_z","gyro_time","gyro_x","gyro_y","gyro_z", ...
            "mag_time","mag_x","mag_y","mag_z","audio_time","audio_data","label","name", ...
            "audio_sample_rate","label_sample_rate","imu_sample_rate","-v7.3")
    end
    fclose(fileID);
end

