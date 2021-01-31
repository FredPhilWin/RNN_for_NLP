import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import utils
import random

'''
ToDos: 
- Enable Batch processing by introducing zero padding 
'''
#Characters that are allowed in the name string, letters that cannot be converted from unicode into one of these ASCII characters will be omitted (see utils.string_to_tensor() for more info)
allowed_letters = " -abcdefghijklmnopqrstuvwxyz"

#Path to directory where the training data is stored.
data_dir_path = "/Users/fredericwinter/Desktop/Programmieren/Python/Pytorch/NLP/Predict_Country_by_Lastname/data/names/"
#Path to directory where parameters / state of the trained model will be saved
save_dir_path = "/Users/fredericwinter/Desktop/Programmieren/Python/Pytorch/NLP/Predict_Country_by_Lastname/model_params/"
#Name of the file in which the moel state will be saved after training
save_name = "RNN_Last_Names_v1" + ".pt"


inference_mode = False #skip training and load saved model state from load_state_path (specify below)
load_state_path = "/Users/fredericwinter/Desktop/Programmieren/Python/Pytorch/NLP/Predict_Country_by_Lastname/model_params/v5_SGD_clipped_gradients.pt"
print_output_after_each_epoch = False #If set to true, the model's classification of the last validation sample in each epoch will be printed out
book_keeping = False #If True, the current progress in each epoch will be printed during training after book_keeping_iters iterations
book_keeping_iters = 1000 #Number of iterations after which a book_keeping entry will be printed

#Load data
names = []
labels = []
label_key = []
for file in os.listdir(data_dir_path):
    file_obj = open(data_dir_path + file)
    names_list = file_obj.read().split()
    names.append(names_list)
    label_key += [file[:-4]]
    labels.append([file[:-4] for name in range(len(names_list))])
    file_obj.close()

label_key.sort()

#stratify names:
class_sizes = [len(labels[i]) for i in range(len(labels))]
max_class = max(class_sizes)

x_train = []
y_train = []
x_val = []
y_val = []
x_test = []
y_test = []
for i in range(len(labels)):
    x_train_i, y_train_i, x_val_i, y_val_i, x_test_i, y_test_i = utils.split_data(names[i], labels[i])
    strat_factor = int(max_class/class_sizes[i])

    for n in range(strat_factor):
        x_train += x_train_i
        y_train += y_train_i
        x_val += x_val_i
        y_val += y_val_i
        x_test += x_test_i
        y_test += y_test_i

#Create RNN
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN,self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) #define fully connected layer to compute hidden state
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.c2c = nn.Linear(input_size + hidden_size, input_size + hidden_size)
        self.o2o = nn.Linear(output_size,output_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)
        self.ReLU = nn.ReLU()
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor),1) #concatenate input tensor and hidden tensor into a combined tensor
        combined = self.ReLU(combined)
        combined = self.c2c(combined)
        combined = self.ReLU(combined)

        hidden = self.i2h(combined)
        hidden = self.ReLU(hidden)
        hidden = self.h2h(hidden)

        output = self.i2o(combined)
        output = self.ReLU(output)
        output = self.o2o(output)

        # Softmax is already included in PyTorch's Cross Entropy Loss module
        # output = self.softmax(output)

        return output, hidden

    def initial_hidden(self):
        return torch.zeros(1, self.hidden_size)

#Set hyperparameters
n_hidden = 128
n_labels = len(label_key)
rnn = RNN(len(allowed_letters), n_hidden, n_labels)

#criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001

optimizer = torch.optim.SGD(rnn.parameters(), learning_rate)
#optimizer = torch.optim.Adam(rnn.parameters(),learning_rate)


def training_step(string, label):
    rnn.train()
    hidden = rnn.initial_hidden()

    for i in range(len(string)):
        output, hidden = rnn(string[i], hidden)

    loss = criterion(output, label)

    optimizer.zero_grad()

    #gradient clipping
    max_norm = 10
    nn.utils.clip_grad_norm_(rnn.parameters(), max_norm)

    loss.backward()
    optimizer.step()

    return output, loss.item()

def validation_step(string, label):
    rnn.eval()
    hidden = rnn.initial_hidden()
    with torch.no_grad():
        for i in range(len(string)):
            output, hidden = rnn(string[i], hidden)

        loss = criterion(output,label)

    return output, loss.item()

if inference_mode == False:
    train_losses = []
    val_losses = []
    max_epochs = 150
    patience = 20
    early_stopping = True

    #helper variables for implementation of early stopping
    trigger = patience  # initialize trigger to the max number of patience
    stopped_early = False

    #Run Model
    for epoch in range(max_epochs):

        #draw samples and convert to tensors
        seed = random.randint(1, 10000)
        random.seed(seed)
        x_train_samples = random.sample(x_train, 20000)
        random.seed(seed)
        y_train_samples = random.sample(y_train, 20000)

        x_train_samples = utils.alphabetical_data_to_tensor(x_train_samples, allowed_letters)
        y_train_samples = utils.labels_to_tensor(y_train_samples)

        # training
        epoch_loss = 0
        for i in range(len(x_train_samples)):
            output, loss = training_step(x_train_samples[i], y_train_samples[i])
            epoch_loss += loss

            if book_keeping and i % book_keeping_iters == 0 and i>0:
                print(f"Epoch {epoch+1}: {(i/len(x_train_samples)*100):0.1f}% completed")

        train_losses.append(epoch_loss / len(x_train_samples))

        #draw validation sample and convert to tensors
        #seed = random.randint(1, 10000)
        seed = 1234
        random.seed(seed)
        x_val_samples = random.sample(x_train, 6000)
        random.seed(seed)
        y_val_samples = random.sample(y_train, 6000)

        x_val_samples = utils.alphabetical_data_to_tensor(x_val_samples, allowed_letters)
        y_val_samples = utils.labels_to_tensor(y_val_samples)

        #validation
        val_loss = 0
        n_correct = 0
        for i in range(len(x_val_samples)):
            output, loss = validation_step(x_val_samples[i], y_val_samples[i])
            val_loss += loss
            prediction = torch.argmax(output, dim=1).item()
            ground_truth = y_val_samples[i].item()

            if prediction == y_val_samples[i]:
                n_correct += 1

        val_acc = n_correct / len(x_val_samples)
        val_losses.append(val_loss / len(x_val_samples))

        print(
            f"Epoch {epoch + 1}/{max_epochs} completed, "
            f"Training Loss: {(epoch_loss / len(x_train_samples)):0.5f}, "
            f"Validation Loss: {(val_loss / len(x_val_samples)):0.5f}, "
            f"Validation Accuracy: {(val_acc * 100):0.2f}%")

        if print_output_after_each_epoch:
            #Print a prediction at the end of each epoch
            name = ""
            for one_hot_letter in x_val_samples[i]:
                one_hot_letter = one_hot_letter[0].tolist()
                letter = allowed_letters[one_hot_letter.index(1)]
                name += letter
            name = name.capitalize()

            if prediction == y_val_samples[i]:
                pred_correct = "CORRECT"
            else:
                pred_correct = "INCORRECT"
            print(f"Name: {name}, prediction {pred_correct}, Predicted: {label_key[prediction]}, Actual: {label_key[ground_truth]}")

        #select best model parameters
        if epoch == 0:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_params = rnn.state_dict()

        elif val_loss <= best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_params = rnn.state_dict()
            trigger = patience

        elif val_loss > best_val_loss or val_loss == "NaN":
            trigger -= 1


        #early stopping, if set to true in line 117
        if early_stopping:
            if trigger == 0:
                print(f"Stopped early after {epoch+1} epochs, "
                      f"Best Model - Validation Loss: {(best_val_loss/len(x_val_samples)):0.5f}, "
                      f"Validation Accuracy: {(best_val_acc*100):0.2f}%\n")
                stopped_early = True
                break

    if not stopped_early:
        print(f"Training completed after {epoch+1} epochs, "
              f"Best Model - Validation Loss: {(best_val_loss/len(x_val_samples)):0.5f}, "
              f"Validation Accuracy: {(best_val_acc*100):0.2f}%\n")

    #plot train and validation loss
    figure = plt.figure()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.show()

    #Testing
    print("Proceed with testing?")
    answer = input()
    if answer.lower() == "yes" or answer.lower() == "y":
        x_test = utils.alphabetical_data_to_tensor(x_test, allowed_letters)
        y_test = utils.labels_to_tensor(y_test)

        for i in range(len(x_val_samples)):
            output, loss = validation_step(x_test[i], y_test[i])
            prediction = torch.argmax(output, dim=1).item()
            ground_truth = y_test[i].item()

            if prediction == y_test[i]:
                n_correct += 1

        test_acc = n_correct / len(x_test)

        print(f"Model accuracy on the test set: {(test_acc*100):0.2f}%")

        # save model state
        if save_dir_path != "":
            print(f"Do you want to save the model parameters to: {save_dir_path}?")
            answer = input()
            if answer.lower() == "yes" or answer == "y":
                torch.save(best_params, save_dir_path + save_name)
                figure.savefig(save_dir_path + save_name + ".png")
                print(f"Best model paramaters were saved to: {save_dir_path}\n")
            else:
                print("Model parameters were not saved.\n")


    print("Proceed with inference?")
    answer = input()
    if answer.lower() == "yes" or answer.lower() == "y":

        best_model = RNN(len(allowed_letters), n_hidden, n_labels)
        best_model.load_state_dict(best_params)

        while True:
            print("Enter a name below or enter 'quit' to terminate")
            name = input().lower()
            if name == "quit" or name == "stop" or name == "exit":
                break
            else:
                pred = utils.infer_language(name, best_model, label_key, allowed_letters)
                print(pred)

#Infere used input names
elif inference_mode == True:

    loaded_model = RNN(len(allowed_letters), n_hidden, n_labels)
    loaded_model.load_state_dict(torch.load(load_state_path))

    while True:
        print("Enter a last name below or enter 'quit' to terminate")
        name = input().lower()
        if name == "quit" or name == "stop" or name == "exit":
            break
        else:
            pred = utils.infer_language(name, loaded_model, label_key, allowed_letters)
            print(pred)