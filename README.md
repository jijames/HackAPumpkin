## HackAPumpkin

TensorFlow pumpkin detector written for the #HackAPumpkin challenge.

### What's the deal?

Apparenly when people put challenges on Twitter, I do them...

### No, what's the deal deal?

I'm learning TensorFlow, and image classification is one of the easier tasks. I thought I would
give it a go with pumpkins.

I started with two classes - carved and not-carved. There are only 50 training images (taken from Google Images).
The validation set only has 20 images. I didn't expect very good results with such little data.

There are two folders containing the training and validation sets. Inside each are sub-folders with each class of image.

Two images are in the top directory that I use for a final prediction. Ideally, the ```pCarved``` image would detect as
a carved pumpkin, and the ```pNotCarved``` would detect as normal. As it is, everything is detected as *normal*.

### Specifics

Take a look at the RunLog.txt. First, you can see the images being loaded, and the image classes detected (folder name).
Next we have our nerual network setup for working with color images. After that is the training phase. Notice the validation
accuracy stays pretty terrible. However, the training accuracy keeps going up! This is a classic case of overfitting.

Let's take a look at the graph of trianing over time...

![Model training accuracy](/foo.png)

You can see that the training accuracy (blue) keeps going up, but the validation stays flat... Not a good sign. Basically,
the system is doing better on it's training data, but continues to be trash on other related data.

It's pretty clear that our model won't work well, so we use some new images and make predictions:

```bash
{'carved': 0, 'normal': 1}
[[-814.74146  360.9489 ]]
[[-712.7013   330.64853]]
```

The first line are the classes. The second line is the prediction for our carved picture. The system thinks the carved pumpkin is normal - awwwww. The third line is a normal pumpkin. The system thinks it's normal (but less confident than the carved one, weird).

Does that mean our model is good for predicting normal pumpkins? Probably not. The model is probably just being lazy and thowing everything
in the normal (non-carved) class.

![The accuracy of the model be like](/pumpkin/carved/34.jpeg)

### Future work?

There are a few ways out of the current badness. I think tweaking the classifier won't produce much better results.
Instead, we should add (a lot) more samples. Alternatively - and what I'll probably actually do - is to create a single-class
classifier. Just Pumpkin? Yes/No.
