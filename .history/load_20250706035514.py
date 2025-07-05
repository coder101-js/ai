🔧 Using 2 CPU threads for training
⚡ Successfully compiled the model with torch.compile()
Traceback (most recent call last):
  File "/root/ai/ai/ai.py", line 155, in <module>
    train()
  File "/root/ai/ai/ai.py", line 73, in train
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: ReduceLROnPlateau.__init__() got an unexpected keyword argument 'verbose'