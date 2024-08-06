# gpt2-style-transfer
Text Style Converter Based on GPT2
Small scale experimental iteration
First time: Set fixed hyperparameters, select a story, train a pre trained model, and only observe the loss value
Second time: Try different hyperparameters, add BLEU evaluation, use mixed precision training, observe loss values
Third time: Use PolynomialDecay learning rate scheduler, add perplexity assessment, expand the training set to three stories, save generated text, and draw learning rate and perplexity curves
Fourth time: Add validation and test sets, adjust each training batch, add GPT2 as a discriminator, and draw loss and accuracy curves
Fifth time: Try adding SAE instead of style classifier and save the results of the middle layer of the model
Sixth time: Optimize model expression

Afterwards, train the entire corpus on a single block V4 for about a day and deploy the model to the front-end and back-end projects
