**See backprop_notes : 1 to 4 images**

**Activation function** : A mathematical function applied to the output of a neuron. It introduces non-linearity into the model.

If a neural networks uses a linear activation function for every layer, regardless of how many layers are used, it is actually no different than running linear regression and defeats the purpose of the neural network. Likewise, using a linear function for all hidden layers and a sigmoid function for the output layer is not different than running logistic regression. 

Examples : sigmoid - slow learner; relu - fast learner; softmax

**Gradient Descent** : 

Tries to move down a gradient (steepest direction) and reach a local minimum (easier than looking for global minimum). That is, compute gradient , take a small step in negative direction of gradient and continue.

Optimizers : SGD , Adam (more common with neural networks), etc

Adam: One optimization to help increase the efficiency of deep learning problems deals with the learning rate $\alpha$ in the gradient descent. It is possible to automatically recognize when the learning rate (thus each step) is too small or too big, and the rate can be adjusted accordingly. This modification is called the Adam Algorithm (Adaptive Moment estimation), and it handles both cases.

The algorithm works by not using one learning rate across all parameters $w$. Every parameter, including $b$, has its own learning rate.

The intuition behind this concept is if $w_j$ or $b$ keeps moving in the same direction (underfitted), $\alpha_j$ should be increased. Conversely, if $w_j$ or $b$ keeps oscillating (overfitted), $\alpha_j$ should be reduced. The details of how this works are a bit complicated, but in simple terms, it keeps track of the moving average of the first and second moment of the gradient.

**Classification vs Regression**

The main difference : Regression algorithms are used to predict continuous values such as price, salary, age, etc. and Classification algorithms are used to predict/classify the discrete values such as Male or Female, True or False, Spam or Not Spam, etc.