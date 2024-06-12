just cat all the xa* files together to get the model.pth file
Github does not allow >100mb files ; these were slpit using `slit -b model_complete.pth`
undo this operation with cat `cat x* > model_complete.pth`
