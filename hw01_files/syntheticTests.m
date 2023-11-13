clear all; close all;

%% Dataset 1
load synthetic1

[w,b] = LDAFit(X,Y);
disp("W")
length(w)
disp("b")
length(b)
plotLinearClassifier(X,Y,w,b,1,221);
title('LDA')

%[w,b] = LRFit(X,Y);
%plotLinearClassifier(X,Y,w,b,1,222);
%title('Logistic regression')

[w,b] = PLAFit(X,Y);
plotLinearClassifier(X,Y,w,b,1,223);
title('PLA')
disp("W")
length(w)
disp("b")
length(b)
%% Dataset 2
load synthetic2

[w,b] = LDAFit(X,Y);
plotLinearClassifier(X,Y,w,b,2,221);
title('LDA')

[w,b] = LRFit(X,Y);
plotLinearClassifier(X,Y,w,b,2,222);
title('Logistic regression')

[w,b] = PLAFit(X,Y);
plotLinearClassifier(X,Y,w,b,2,223);
title('PLA')

%% Dataset 3
load synthetic3

[w,b] = LDAFit(X,Y);
plotLinearClassifier(X,Y,w,b,3,221);
title('LDA')

[w,b] = LRFit(X,Y);
plotLinearClassifier(X,Y,w,b,3,222);
title('Logistic regression')

[w,b] = PLAFit(X,Y);
plotLinearClassifier(X,Y,w,b,3,223);
title('PLA')

%% Dataset 4
load synthetic4

[w,b] = LDAFit(X,Y);
plotLinearClassifier(X,Y,w,b,4,221);
title('LDA')

[w,b] = LRFit(X,Y);
plotLinearClassifier(X,Y,w,b,4,222);
title('Logistic regression')

[w,b] = PLAFit(X,Y);
plotLinearClassifier(X,Y,w,b,4,223);
title('PLA')
