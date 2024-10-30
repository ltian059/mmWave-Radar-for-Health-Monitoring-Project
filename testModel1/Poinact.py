import torch
import torch.nn as nn
import torch.optim as optim
from PointNet_lstm import HAR_model
from torch.optim.lr_scheduler import StepLR
import torch.utils.data as Data
# import matplotlib.pyplot as plt
# import matplotlib
import numpy as np
# from data import process_data,radhar_data,test_data,radhar_data3
import pickle
import numpy as np
#from torchsummary import summary
from torchinfo import summary
from torch.utils.data import random_split
import random
import os
from datetime import datetime

###BETTER TO RUN THIS CODE USING NOHUP COMMAND###########


#####========================================================######
#####========================================================######
##REMEMBER TO EDIT THE DICTIONARY BELOW.##############
def to_class_accuracy_dict(list):
    dict = {
    "Falling": list[0],
    "LayBed": list[1],
    "LayFloor": list[2],
    "Sitting": list[3],
    "Stand_Walking": list[4]
    }
    return dict
window_length = 20
num_cls = 5
read_folder = r'/pointact_for_PCHAR/data_reza_4d_xyzv_2lays_1031-600-600-600-600'
save_folder = r'/pointact_for_PCHAR/saved_model_24.10.21.reza_xyzv_2lays_1031-600-600-600-600'
num_epochs = 500
dropout_rate = 0.1
learning_rate = 0.0001
weight_decay = 0.001
input_channels = 4
train_ratio = 0.85
feature_transform = False
#####========================================================######
#####========================================================######
print(f'dataset read from: {read_folder}')
print(f'model will be saved to: {save_folder}')
print('starting time: '+ str(datetime.now()))
print(f'window length: {window_length}')
print(f'number of classes: {num_cls}')
print(f'dropout rate: {dropout_rate}')
print(f'number of epochs: {num_epochs}')
print(f'learning rate: {learning_rate}')
print(f'weight decay variable: {weight_decay}')
print(f'number of input channels: {input_channels}')
print(f'train data : (train data + test data) = {train_ratio}')
print(f'whether feature transform: {feature_transform}')

device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'device used for training: {device}')

# load traindata
# change the paths
'''
with open(r'/pointact_for_PCHAR/front_zero_train_data.pkl', 'rb') as file:
    traindata = pickle.load(file)
print(traindata.size())
with open(r'/pointact_for_PCHAR/front_zero_test_data.pkl', 'rb') as file:
    testdata = pickle.load(file)
'''
input_path = os.path.join(read_folder, 'input.pkl')
label_path = os.path.join(read_folder, 'label.pkl')

with open(input_path, 'rb') as file:  # input of model
    wholedata = pickle.load(file)
print(f'input pickle size{wholedata.size()}')


with open(label_path, 'rb') as file:  #label of model
    truthdata = pickle.load(file)
print(f'output pickle size{truthdata.size()}')

# 初始化模型和损失函数、优化器
# 定义模型和损失函数
model = HAR_model(output_dim=num_cls, frame_num=window_length, input_channels=input_channels, dropout_rate=dropout_rate, feature_transform=feature_transform)  # 根据您的模型定义进行实例化，这里假设模型名称为HAR_model
criterion = nn.CrossEntropyLoss()
print(model)


# 定义训练数据和真实结果  已废弃
#truthdata = torch.tensor([])

all_classes = [0]*num_cls
for i in range(truthdata.shape[0]):
    label = truthdata[i].item()
    label = int(label)
    all_classes[label] += 1
print('distribution of all data:')
print(to_class_accuracy_dict(all_classes))

# 定义优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  #change the learning rate
# optimizer = optim.SGD(model.parameters(), lr=0.0001)
'''
traindata = traindata.to(device)
testdata = testdata.to(device)
truthdata_tensor = torch.tensor(truthdata).to(device)
truthdata_test_tensor = torch.tensor(truthdata_test).to(device)
'''
#wholedata = wholedata.to(device)
#truthdata = truthdata.to(device)

model = model.to(device)
print(wholedata.shape)
print(truthdata.shape)

'''
training_set = Data.TensorDataset(traindata, truthdata_tensor)
testing_set = Data.TensorDataset(testdata, truthdata_test_tensor)
loader = Data.DataLoader(dataset=training_set, batch_size=16, shuffle=False)  #change the batch size
loader_test = Data.DataLoader(dataset=testing_set, batch_size=16, shuffle=False)   #change the batch size
'''

whole_dataset = Data.TensorDataset(wholedata, truthdata)


train_size = int(train_ratio * wholedata.shape[0])
test_size = wholedata.shape[0] - train_size

training_set, testing_set = random_split(whole_dataset, [train_size, test_size])



train_classes = [0]*num_cls
test_classes = [0]*num_cls
for x, y in training_set:
    y = y.item()
    y = int(y)
    train_classes[y]+=1
for x, y in testing_set:
    y = y.item()
    y = int(y)
    test_classes[y]+=1
print('distribution of train data:')
print(to_class_accuracy_dict(train_classes))
print('distribution of test data:')
print(to_class_accuracy_dict(test_classes))

print(len(training_set))
print(len(testing_set))
'''
train_size1 = int(train_size/2)
train_size2 = train_size - train_size1
training_set1, training_set2 = random_split(training_set, [train_size1, train_size2])
print(len(training_set1))
print(len(training_set2))
print(len(testing_set))
loader1 = Data.DataLoader(dataset=training_set1, batch_size=16, shuffle=True, num_workers=4)  #change the batch size
loader2 = Data.DataLoader(dataset=training_set2, batch_size=16, shuffle=True, num_workers=4)  #change the batch size
'''
loader = Data.DataLoader(dataset=training_set, batch_size=25, shuffle=True, num_workers=4)
loader_test = Data.DataLoader(dataset=testing_set, batch_size=16, shuffle=True, num_workers=4)   #change the batch size

# def set_bn_eval(m):
#     if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
#         m.eval()

# def set_bn_train(m):
#     if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
#         m.train()

# def disable_dropout(m):
#     if isinstance(m, nn.Dropout):
#         m.p = 0

# def enable_dropout(m):
#     if isinstance(m, nn.Dropout):
#         m.p = 0.2

# 训练模型

highest_acc = 0
highest_fall_acc = 0
accuracies = []
model.train()
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    for step, (batch_x, batch_y) in enumerate(loader):
        #model.apply(set_bn_train)
        #model.apply(enable_dropout)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        output = model(batch_x)
        # print(f'output size{output.size()}')
        # print(f'batch_y size{batch_y.size()}')
        # print(f'squeezed batch_y size {batch_y.to(torch.int64).t().squeeze().size()}')
        # loss = criterion(output, batch_y.to(torch.int64).t().squeeze())  
        loss = criterion(output, batch_y.to(torch.int64))  
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        _, predicted_labels = torch.max(output, 1)
        correct_predictions = (predicted_labels == batch_y.squeeze()).sum().item()
        accuracy = correct_predictions / batch_y.shape[0]
        #print(accuracy)
        accuracies.append(accuracy)  # 将精度值添加到列表中
        torch.cuda.empty_cache()

    #print(accuracies)
    print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, loss.item()))
    print('accuracy for this epoch: {}'.format(np.mean(accuracies)))
    accuracies = []

    model.eval()  # 将模型设置为评估模式
    test_accs = []
    class_correct = [0] * num_cls  # 用于统计每个类别的正确预测数量
    class_total = [0] * num_cls  # 用于统计每个类别的样本总数

    for step, (batch_x, batch_y) in enumerate(loader_test):
        with torch.no_grad():
            #model.apply(set_bn_eval)
            #model.apply(disable_dropout)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            output_test = model(batch_x)
            _, predicted_labels_test = torch.max(output_test, 1)
            #print(output_test)
            #print('-------')
            #print(predicted_labels_test)
             # 计算每个类别的正确预测数量和样本总数
            for i in range(len(batch_y)):
                label = batch_y[i].item()
                label = int(label)
                class_total[label] += 1
                if int(predicted_labels_test[i]) == int(label):
                    class_correct[label] += 1

            #correct_predictions_test = (predicted_labels_test == batch_y.squeeze(1)).sum().item()
            correct_predictions_test = (predicted_labels_test == batch_y).sum().item()
            accuracy_test = correct_predictions_test / batch_y.shape[0]
            test_accs.append(accuracy_test)
            torch.cuda.empty_cache()

    print('====Total Test Accuracy: {}'.format(np.mean(test_accs)))
    
    if np.mean(test_accs) > highest_acc:
        save_path = save_folder+'/epoch'+str(epoch+1)+'.pt'
        torch.save(model,save_path)
        print('NEW HIGH ACCURACY')
        print('model saved to '+save_path)
        highest_acc = np.mean(test_accs)
    elif (epoch+1)%30 == 0:
        save_path = save_folder+'/epoch'+str(epoch+1)+'.pt'
        torch.save(model,save_path)
        print('model saved to '+save_path)
    correct_arr = np.array(class_correct)
    total_arr = np.array(class_total)
    accuracy_arr = list(correct_arr/total_arr)
    print(correct_arr)
    print(total_arr)
    print(accuracy_arr)
    if accuracy_arr[0] > highest_fall_acc:
        save_path = save_folder+'/epoch'+str(epoch+1)+'.pt'
        torch.save(model,save_path)
        print('NEW HIGH FALL ACCURACY')
        print('model saved to '+save_path)
        highest_fall_acc = accuracy_arr[0]
    print('Each Class\'s Test Accuracy:')
    print(to_class_accuracy_dict(accuracy_arr))
    print('===============================================================================')
    print('========================= END OF EPOCH'+str(epoch+1) +' =====================================')
    print('===============================================================================')
    torch.cuda.empty_cache()




#filename = 'accuracies2.txt'
#np.savetxt(filename, accuracies)

# torch.save(model.state_dict(), 'model.pth')
#------------------------------------------------------------
# 测试模型

'''
model.eval()  # 将模型设置为评估模式

with torch.no_grad():
    output_test = model(traindata)
    _, predicted_labels_test = torch.max(output_test, 1)
    correct_predictions_test = (predicted_labels_test == truthdata.squeeze(1)).sum().item()
    accuracy_test = correct_predictions_test / traindata.shape[0]

print('Test Accuracy: {:.2f}%'.format(accuracy_test * 100))
'''