# PyTorch training loop

This PyTorch training loop aims to cover all the fundamentals of what you need in a training loop, while still offering customizability. Since all of the code is done in one function, you can customize every part of the training process. It is also very simple to use and understand. As a bonus, there is also a progress bar included with the training loop. This progress bar is also as customizable as the training loop, and doesn't just have to be used for this training loop. Using this progress bar, you can print out the percentage as well as any statistics throughout the training process.

## Usage

There are two files with one containing the progress bar(`pbar.py`) and the other containing the trianing loop(`trainloop.py`). You can either copy the code from each file and use accordingly, or you can use straight from the files. The required libraries are imported at the top of each file, excluding the PyTorch imports. In `trainloop.py`, the function with all the training code requires a number of parameters. These are the ones that you create and put into the function when training. There are no defaults, so everything has to be decided by you.

## Limitations

Although this is a very useful tool for a lot of deep learning practitioners using PyTorch, there are a few limitations to this code. 

1. The training loop does not offer support for callbacks. Anything that you need to add to the training process has to be separately added in. 
2. Training stats vary, such as the built in support for accuracy. If you're doing something even slightly irregular, you'll probably have to remove all the code having to do with the accuracy and add new code(or just stick with loss).

## License

See [LICENSE](https://github.com/totoys/pytorch-training-loop/blob/master/LICENSE)