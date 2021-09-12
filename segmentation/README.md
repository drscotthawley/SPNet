# SPNet Segmentation

## Strategy:

- [x] Run / follow [original fastai segmentation tutorial](https://docs.fast.ai/tutorial.vision.html) 
- [x] and [Zach's WWF Segmentation Lesson](https://walkwithfastai.com/Segmentation) 
- [x] Run / follow [Zach's WWF tutorial on Hybridizing Models](https://walkwithfastai.com/Hybridizing_Models) with UNet 
- [ ] Re-do the original segmentation example BUT with one floating-point class, as follows:
  - [ ] use MSELoss and modify the Dataloader to do 1 (float) class instead of multiple (int) classes
  - [ ] modify the model head as per Zach's Hybridizing tutorial
  - [ ] Run it and see how accuracy (or other metrics?) compare to original version
- [ ] Dig out / rewrite my script for producing segmentation masks from ellipses. Use fake / CycleGAN data.
- [ ] Modify Dataloader again to be able to *read* floats 
- [ ] Train on fake / CycleGAN / real data
- [ ] Predict on images of guitars & stuff 
- [ ] Write the paper 

...Meanwhile Grant et al have been cleaning data.  

- [ ] Regenerate mask data, repeat training.  Fill in paper graphs

- [ ] Submit and profit

  
