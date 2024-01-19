# MMER-TD
With technological advancements, we can now capture rich dialogue content, tones, textual information, and visual data through tools like microphones, the internet, and cameras. However, relying solely on a single modality for emotion analysis often fails to accurately reflect the true emotional state, as this approach overlooks the dynamic correlations between different modalities. To address this, our study introduces a multimodal emotion recognition method that combines tensor decomposition fusion and self-supervised multi-task learning. This method first employs Tucker decomposition techniques to effectively reduce the model’s parameter count, lowering the risk of overfitting. Subsequently, by building a learning mechanism for both multimodal and unimodal tasks and incorporating the concept of label generation, it more accurately captures the emotional differences between modalities. We conducted extensive experiments and analyses on public datasets like CMU-MOSI and CMU-MOSEI, and the results show that our method significantly outperforms existing methods in terms of performance.

Here is a visual representation of the model architecture used in this project.
<div align=center>
<img src="https://github.com/ZhuJw31/MMER-TD/blob/main/structure/Framework.png">
</div>

*__Experimental Results and Visualization__*

This section presents some of the experimental results obtained with the model, along with visualizations that shed light on its performance.
<div align=center>
<img src="https://github.com/ZhuJw31/MMER-TD/blob/main/tables/1.png">  
  
<img src=https://github.com/ZhuJw31/MMER-TD/blob/main/Visualization/5final.png width="50%" height="50%">
  

![7](https://github.com/ZhuJw31/MMER-TD/assets/76214301/85455e39-611b-4f0e-859b-17ea91faf5e3)
</div>
