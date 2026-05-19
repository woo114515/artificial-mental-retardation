'''
训练入口。不负责具体细节，只把其他模块串起来。
'''

from data.dataloader import create_mini_batches
import nn
from config import *
import numpy as np

np.random.seed(RANDOM_SEED)

# load MNIST
from data.mnist import X_train, y_train, X_val, y_val, X_test, y_test

# build model from config, e.g. 784 -> 128 -> 10
layers = []
layer_dims = [INPUT_DIM] + HIDDEN_DIMS + [NUM_CLASSES]

for layer_index in range(len(layer_dims) - 1):
    layers.append(
        nn.Linear(
            in_features=layer_dims[layer_index],
            out_features=layer_dims[layer_index + 1],
            method=WEIGHT_INIT,
            seed=RANDOM_SEED + layer_index,
        )
    )

    if layer_index < len(layer_dims) - 2:
        layers.append(nn.ReLU())

model = nn.Sequential(layers)
criterion = nn.SoftmaxCrossEntropyLoss()
optimizer = nn.SGD(LEARNING_RATE)

# train
'''
for epoch in range(NUM_EPOCHS):
    for X_batch, y_batch in create_mini_batches(...):
        forward
        loss
        backward
        update
    evaluate train / val
'''

for epoch in range(NUM_EPOCHS):
    for X_batch, y_batch in create_mini_batches( X_train, y_train, batch_size=BATCH_SIZE):
        
        logits = model.forward(X_batch)
        loss = criterion.forward(logits, y_batch)
        
        dlogits = criterion.backward()
        model.backward(dout=dlogits)
        
        params_and_grads = model.params_and_grads()
        optimizer.step(params_and_grads)
    
    val_logits = model.forward(X_val)
    val_loss = criterion.forward(val_logits, y_val) # loss
    val_predict = np.argmax(val_logits, axis=1)
    val_accuracy = np.mean(val_predict == y_val)# accuracy
    print(f"Epoch {epoch + 1}: loss={val_loss:.4f}, accuracy={val_accuracy:.4f}")

# eval
test_logits = model.forward(X_test)
test_accuracy = np.mean(np.argmax(test_logits, axis=1) == y_test)
print(f"Final test accuracy={test_accuracy}")
