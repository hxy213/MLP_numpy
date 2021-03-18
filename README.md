# MLP

#### Dependencies

- Python 3.8

- numpy

- random

- os

- matplotlib

#### Forward propagation 

![image](https://github.com/hxy213/MLP_numpy/blob/main/figures/Forward.png)

#### Loss function

![image](https://github.com/hxy213/MLP_numpy/blob/main/figures/LossFunction.png)

#### Stochastic gradient descent



#### How to use this model 

```python
# Initialize w ,learning_rate
model = MLP_model.MLP_Classification(n_features, n_hidden,
                     n_classes, learning_rate)

# Train the model x(1,n_features) label(1,n_classes)
# Use loop to take all samples for training
loss_ = model.train( x, label)

# Test, calculate the accuracy of the data  x(n,n_features) label(n,n_classes)
accuracy = model.accuracy(x, label)
```

```python
# save model
path = os.path.join('MLP_Parameters/model%s/' % (crossValidation))
folder = os.getcwd() +'/'+ path
if not os.path.exists(folder):
    os.makedirs(folder)
model.save(path)

# load model
model = MLP_model.MLP_Classification(n_features, n_hidden,
                                         n_classes, learning_rate)
model.load(path)
```