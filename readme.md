# OpenBCI ML Implementation Capstone Project

## Comments for Backpropagation.py module:

Updates:

  ### @Eric 
    - updated the other NN modules (utilizing the Convolutional method)
    - added the spider robot contorl script (QuadBotTestV3.ino)

  ### @zoot-io
    - updated StoreData.py
      - no more side jobs (data trimming or filtering)
      - increased data collection time to 10 seconds
    - Check FullData for full data points (10 second runs)
    - Check UnreliableData for, well, my mistakes... 
      - Those runs weren't grounded.
    - Check TrimmedData for trimmed data points (5 second runs)
    
  ### @ayyymiel
    - made backprop dataset
    - edited backprop to take in the dataset
      > overflow errors in activation function
    To-Do:
      > fix overflow errors, try new activation func
      > move function comments from old backpropagation module to new
