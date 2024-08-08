# gpt2-style-transfer
Text Style Converter Based on GPT2
Small scale experimental iteration

First time: Set fixed hyperparameters, select a story, train a pre trained model, and only observe the loss value

Second time: Try different hyperparameters, add BLEU evaluation, use mixed precision training, use PolynomialDecay learning rate scheduler, add perplexity evaluation, expand the training set to three stories, save generated text, draw learning rate and perplexity curves

Third time: Add validation and test sets, adjust each training batch

Fourth time: Change the mindset. First, divide the original dataset into two styles x and y, implement style x to y conversion, add vx and vy to represent style 
label embedding, implement discriminator y and generator, add discriminator z to align the distribution of content vector z

Fifth time: Use KL divergence regularization on the z-distribution and optimize the model expression using the top_k activation function

Sixth time: Enhance the training set with data and try to generalize the model to any style of input text that can convert style into y

Afterwards, train the entire corpus on a single block V4 for about a day and deploy the model to the front-end and back-end projects
