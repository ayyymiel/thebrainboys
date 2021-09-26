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
    - Created correlation module to generate correlation matrix to analyze interaction between nodes
    - Uploaded screenshots from plots per "thought"
    - Uploaded separate file for backpropagation prediction functions
    To-Do:
      > Analyze correlation plots
      > Train backprop model on trimmed dataset by @Eric
