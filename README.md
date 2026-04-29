# Image Captioning with Attention + Grad-CAM

End-to-end implementation of an attention-based image-captioning model on Flickr8k,
with Grad-CAM visual explanations and a Streamlit web demo.

**Status:** in development — Day 1 scaffold.

## Stack
- PyTorch 2.x, torchvision
- Streamlit (web app)
- NLTK (BLEU evaluation)

## Project structure
```
src/        # model, dataset, training, inference, Grad-CAM
app/        # Streamlit web app
notebooks/  # exploration + visualization
```

## References
1. Bahdanau, Cho, Bengio. *Neural Machine Translation by Jointly Learning to Align and Translate.* ICLR 2015.
2. Xu et al. *Show, Attend and Tell.* ICML 2015.
3. Selvaraju et al. *Grad-CAM.* ICCV 2017.

## License
MIT
