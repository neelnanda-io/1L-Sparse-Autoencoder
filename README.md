# TLDR

This is an open source replication of [Anthropic's Towards Monosemanticity](https://transformer-circuits.pub/2023/monosemantic-features/index.html) paper. The autoencoder was trained on the gelu-1l model in TransformerLens, you can access two trained autoencoders and the model using [this tutorial](https://colab.research.google.com/drive/1u8larhpxy8w4mMsJiSBddNOzFGj7_RTn#scrollTo=MYrIYDEfBtbL). 

# Reading This Codebase

This is a pretty scrappy training codebase, and won't run from the top. I mostly recommend reading the code and copying snippets. See also [Hoagy Cunningham's Github](https://github.com/HoagyC/sparse_coding).

* `utils.py` contains various utils to define the Autoencoder, data Buffer and training data. 
  * Toggle `loading_data_first_time` to True to load and process the text data used to run the model and generate acts
* `train.py` is a scrappy training script
  * `cfg["remove_rare_dir"]` was an experiment in training an autoencoder whose features were all orthogonal to the shared direction among rare features, those lines of code can be ignored and weren't used for the open source autoencoders. 
  * There was a bug in the code to set the decoder weights to have unit norm - it makes the gradients orthogonal, but I forgot to *also* set the norm to be 1 again after each gradient update (turns out a vector of unit norm plus a perpendicular vector does not remain unit norm!). I think I have now fixed the bug. 
* `analysis.py` is a scrappy set of experiments for exploring the autoencoder. I recommend reading the Colab tutorial instead for something cleaner and better commented. 

Setup Notes:

* Create data - you'll need to set the flag loading_data_first_time to True in utils.py , note that this downloads the training mix of gelu-1l and if using eg the Pythia models you'll need different data (I recommend https://huggingface.co/datasets/monology/pile-uncopyrighted )
* A bunch of folders are hard coded to be /workspace/..., change this for your system.
* Create a checkpoints dir in /workspace/1L-Sparse-Autoencoder/checkpoints

* If you train an autoencoder and want to share the weights, copy the final checkpoints to a new folder, use upload_folder_to_hf to upload to HuggingFace, create your own repo. Run huggingface-cli login to login, and apt-get install git-lfs and then git lfs install