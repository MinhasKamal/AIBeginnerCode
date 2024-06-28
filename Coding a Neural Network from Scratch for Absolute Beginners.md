Have you heard of rice milling machine? Well, here is an image of it-

![1_PCNHO0sTPOwA18By6p9BMQ](https://github.com/MinhasKamal/AIBeginnerCode/assets/5456665/4a641163-3b1c-4ba9-98eb-392f4f0e65be)

*This article is pretty long. But this is not to waste your time reading. This long writing is actually here for making things more intuitive and speeding up your learning. Also, I will avoid technical terms and math to make things simpler to consume.*

Great! Lets come back to the rice milling machine. So, this machine take in paddies and then does some processing (threshing). At the end, you will get rice and husk as output-

![1_2qSuSwsJzLlHL80RUPYr9Q](https://github.com/MinhasKamal/AIBeginnerCode/assets/5456665/66418424-74af-4f4a-8cf7-d857836adb7d)

Neural Networks or any AI model in general are like processing machines that take some data as input, then runs some kind of transformation, and finally output results. For example, if you provide a goat  image to the model, then it will transform the image into the word "goat".

Lets jump into a simple problem: predicting storm. If we have dark clouds and a sudden drop in temperature, then we predict  that a storm is coming. Here is a tabular representation of the data-

![1_TIl5bNmlQeKeoOkISW72NA](https://github.com/MinhasKamal/AIBeginnerCode/assets/5456665/690942ac-2ad8-400d-8ddb-357ed97be60d)

In real life, weather prediction is a complex process. Here, we are making this simple scenario where, we shall predict that a storm will happen only if we have both dark clouds and a temperature decrease. Now, we can solve this problem with plain 'if/else'. Here is a python code-

```
def predict(dark_clouds, temperature_drop):
    storm = 0
    if dark_clouds == 1 and temperature_drop == 1:
        storm = 1
    return storm
 
print(predict(1, 1))
print(predict(1, 0))
print(predict(0, 1))
print(predict(0, 0))
```

But, we are doing machine learning here; we do not want to explicitly solve a problem ("Why?", you might ask. Well, there are problems that are extremely hard to program explicitly. And, we use machine learning for solving those problems.). Instead, we should write codes that can evaluate a given solution and then can suggest a better solution-

![1_itmi_oBBInInevageKOOxw](https://github.com/MinhasKamal/AIBeginnerCode/assets/5456665/1ff3bea5-28d2-4778-91e8-c4221d9c6b9a)

In practice, we start with a random prediction and iteratively improve on that. All is good, but what is a neuron?! Neurons are like functions that take in multiple inputs, do some magic, and output the result-

```
def predict(dark_clouds, temperature_drop):
    storm = MAGIC!
    return storm
```

Grander the magic sillier  the secret!! The neuron simply puts weights on each input depending on the input's effect on the output. Then, it accumulates all the weighted inputs. If the sum is greater than 1 then we shall output 1, else 0.

![1_rLhemTn48RMeJXXfIH3nNQ](https://github.com/MinhasKamal/AIBeginnerCode/assets/5456665/c47f6c66-38bd-47d2-b522-1149018daf1d)

As we have two inputs here, we need to keep two weights for them. So, we can create a neuron class with the necessary weights and the predict function.

```
class Neuron:
    def __init__(self):
        self.w1 = 2
        self.w2 = 0.5

    def predict(self, dark_clouds, temperature_drop):
        storm = 0
        if (dark_clouds*self.w1 + temperature_drop*self.w2) > 1:
            storm = 1
        return storm

neuron = Neuron()
print(neuron.predict(1, 1))
print(neuron.predict(1, 0))
print(neuron.predict(0, 1))
print(neuron.predict(0, 0))
```

Wait! Where do we get the values of the weights? Well, that is machine learning :) 

Please, do not just read the article. You will not be able to build a deep intuition just by reading; you need practice. If you have not done it already, open your python editor or GoogleColab and run these codes yourself.

For now, lets set w1=2 and w2=0.5. If you run the code, you will get the following output-
```
1
1
0
0
```

We can see that there is only one mistake here- the 2nd prediction should be 0. Lets set both the weights to 3. So, w1=3 and w2=3. Now, we get this-

```
1
1
1
0
```

Oops! We have two mistakes now- 2nd and 3rd prediction. Lets set w1=0.6 and w2=0.8-

```
1
0
0
0
```

Wow! We have all correct answers now. You can play with different other weight values and check that give correct results.

By now, we do understand that- simply by changing the weights we can adapt our prediction function for any input-output patterns. Therefore, we need a learning function that takes the two inputs and the output, and does some magic (again)  to change the weights accordingly.

Lets initialize the weights with some random values (The predict function is minimized here under […])-

```
import random

class Neuron:
    def __init__(self):
        self.w1 = random.random()
        self.w2 = random.random()

    def predict(self, dark_clouds, temperature_drop): [...]

    def learn(self, dark_clouds, temperature_drop, storm):
        self.w1 = MAGIC!
        self.w2 = MAGIC!
```

So, the magic here is that- first we try to predict the result with the random weights that we have. Then, we calculate error by subtracting the prediction with the actual result. Finally, we update the weights by the error and the related input (that was multiplied by the weight). It is like- penalizing the weights based on their impact (related input)  on the error. You can also think of it like- distributing the error over the weights depending  on their part in generating the error.

```
import random

class Neuron:
    def __init__(self): [...]

    def predict(self, dark_clouds, temperature_drop): [...]

    def learn(self, dark_clouds, temperature_drop, storm):
        error = self.predict(dark_clouds, temperature_drop) - storm
        self.w1 -= error * dark_clouds / 100
        self.w2 -= error * temperature_drop / 100
```

We do not want to change the weights too much at a time; we wish to take small steps towards the solution (big steps often cause divergence). So, we divide the error by 100 while updating the weights.

Now, we can add training and testing code-

```
import random

class Neuron: [...]

neuron = Neuron()

while True:
    # testing
    if (neuron.predict(1, 1) == 1 and
            neuron.predict(1, 0) == 0 and
            neuron.predict(0, 1) == 0 and
            neuron.predict(0, 0) == 0):
        break

    # training
    neuron.learn(1, 1, 1)
    neuron.learn(1, 0, 0)
    neuron.learn(0, 1, 0)
    neuron.learn(0, 0, 0)

# output
print(neuron.predict(1, 1))
print(neuron.predict(1, 0))
print(neuron.predict(0, 1))
print(neuron.predict(0, 0))
```

As you can see, we are running training as long as we get all correct outputs.

![1_3i0cmIcWYGlFjm8iwzhjVA](https://github.com/MinhasKamal/AIBeginnerCode/assets/5456665/60d5ff7f-be27-4261-a35d-fb490efd700f)

Congratulations on reaching this far! You have successfully coded an artificial neuron from scratch and trained it!! Awesome!!!

Lets make functions for training and testing-

```
import random

class Neuron: [...]

data = [[1, 1, 1],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]]

def runTraining(neuron):
    for row in data:
        neuron.learn(row[0], row[1], row[2])

def runTesting(neuron):
    return [neuron.predict(row[0], row[1]) for row in data]

neuron = Neuron()
while True:
    output = runTesting(neuron)
    print(output)
    if output == [row[2] for row in data]:
        break

    runTraining(neuron)
```

If you run the code multiple times, each time you will need different number of steps to reach the solution. This is because of the random values used for initiating the weights. You might even get to solution with only one step. 

Now, lets play with different outputs. Lets change the second output to 1 and run the code multiple times-

```
data = [[1, 1, 1],
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0]]
```

Is it taking more steps to reach the solution? Why that might be the case? Take sometime, think about it. You can print the weights after each training phase and analyze the gradual change. 

Now, lets set the first output to 0 and run again-

```
data = [[1, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
        [0, 0, 0]]
```

Is it taking less steps to reach the solution? Or even more steps? This understanding is very useful to grasp the underlying mechanism of artificial neurons.

Now, lets set all the outputs to 1-

```
data = [[1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1]]
```

What is happening? Did your computer crush? Or, not responding? Actually, it is stuck in the while loop. Because, we never get all correct output. The problem happens for the 4th row- [0, 0, 1]. And, its is easy to understand why- the input values are both 0 (zero). So, no matter what the weights are, zero multiplied by any value is zero. Thus, the condition in the predict() function [(dark_clouds*self.w1 + temperature_drop*self.w2) > 1] will never satisfy.

Stop here, do not read the article any further. Think about this issue we are stuck at. What would you do to solve it? Give your brain at least a minute to digest before going further.

Here is the full code-

```
import random

class Neuron:
    def __init__(self):
        self.w1 = random.random()
        self.w2 = random.random()
        self.t = random.random()

    def predict(self, dark_clouds, temperature_drop):
        storm = 0
        if (dark_clouds*self.w1 + temperature_drop*self.w2) > self.t:
            storm = 1
        return storm

    def learn(self, dark_clouds, temperature_drop, storm):
        error = self.predict(dark_clouds, temperature_drop) - storm
        self.w1 -= error * dark_clouds / 100
        self.w2 -= error * temperature_drop / 100
        self.t += error / 100

data = [[1, 1, 1],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 1]]

def runTraining(neuron):
    for row in data:
        neuron.learn(row[0], row[1], row[2])

def runTesting(neuron):
    return [neuron.predict(row[0], row[1]) for row in data]

neuron = Neuron()
while True:
    output = runTesting(neuron)
    print(output)
    if output == [row[2] for row in data]:
        break

    runTraining(neuron)
```

Now, lets learn some mouthful AI terms. Gradient Descent. Learning Rate. Bias. 

The last test :D Set the values like the following-

```
data = [[1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [0, 0, 0]]
```
