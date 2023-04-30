Download Link: https://assignmentchef.com/product/solved-homework-2-cs-260-machine-learning-algorithms
<br>
<h1>1           Naive Bayes</h1>

The binary Naive Bayes classifier has interesting connections to the logistic regression classifier. You will show that, under certain assumptions, the Naive Bayes likelihood function is identical in form to the likelihood function for logistic regression. You will then derive the MLE parameter estimates under these assumptions.

<ol>

 <li>Suppose <em>X </em>= {<em>X</em><sub>1</sub><em>,…,X<sub>D</sub></em>} is a continuous random vector in R<em><sup>D </sup></em>representing the features and <em>Y </em>is a binary random variable with values in {0<em>,</em>1} representing the class labels. Let the following assumptions hold:

  <ul>

   <li>The label variable <em>Y </em>follows a Bernoulli distribution, with parameter <em>π </em>= <em>P</em>(<em>Y </em>= 1).</li>

   <li>For each feature <em>X<sub>j</sub></em>, we have <em>P</em>(<em>X<sub>j</sub></em>|<em>Y </em>= <em>y<sub>k</sub></em>) follows a Gaussian distribution of the form N(<em>µ<sub>jk</sub>,σ<sub>j</sub></em>).</li>

  </ul></li>

</ol>

Using the Naive Bayes assumption that states “<em>for all j</em><sup>0 </sup>6= <em>j, X<sub>j </sub>and X<sub>j</sub></em>0 <em>are conditionally independent given Y </em>”, compute <em>P</em>(<em>Y </em>= 1|<em>X</em>) and show that it can be written in the following form:

<em>.</em>

Specifically, you need to find the explicit form of <em>w</em><sub>0 </sub>and <strong>w </strong>in terms of <em>π</em>, <em>µ<sub>jk</sub></em>, and <em>σ<sub>j</sub></em>, for <em>j </em>= 1<em>,…,D </em>and <em>k </em>∈ {0<em>,</em>1}.

<ol>

 <li>Suppose a training set with <em>N </em>examples (<strong>x</strong><sub>1</sub><em>,y</em><sub>1</sub>)<em>,</em>(<strong>x</strong><sub>2</sub><em>,y</em><sub>2</sub>)<em>,</em>·· <em>,</em>(<strong>x</strong><em><sub>N</sub>,y<sub>N</sub></em>) is given, where <strong>x</strong><em><sub>i </sub></em>= (<em>x<sub>i</sub></em><sub>1</sub><em>,</em>··· <em>,x<sub>iD</sub></em>)<sup>&gt; </sup>is a <em>D</em>-dimensional feature vector, and <em>y<sub>i </sub></em>∈ {0<em>,</em>1} is its corresponding label. Using the assumptions in 1.1 (not the result), provide the maximum likelihood estimation for the parameters of the Naive Bayes with Gaussian assumption. In other words, you need to provide the estimates for <em>π</em>, <em>µ<sub>jk</sub></em>, and <em>σ<sub>j</sub></em>, for <em>j </em>= 1<em>,…,D </em>and <em>k </em>∈ {0<em>,</em>1}.</li>

</ol>

<h1>2           Logistic Regression</h1>

Consider a binary logistic regression model, where the training samples are <em>linearly separable</em>.

<ol>

 <li>Given <em>n </em>training examples (<strong>x</strong><sub>1</sub><em>,y</em><sub>1</sub>)<em>,</em>(<strong>x</strong><sub>2</sub><em>,y</em><sub>2</sub>)<em>,…,</em>(<strong>x</strong><em><sub>n</sub>,y<sub>n</sub></em>) where <em>y<sub>i </sub></em>∈ {0<em>,</em>1}, write down the negative log likelihood, L(<strong>w</strong>), in terms of the sigmoid function, <em>x</em>, and <em>y</em>.</li>

 <li>Is this loss function convex? Provide your reasoning.</li>

 <li>Show that the magnitude of the optimal <strong>w </strong>can go to infinity when the training samples are <em>linearly separable</em>.</li>

 <li>A convenient way to prevent numerical instability issues is to add a penalty term to the likelihood function as follows:</li>

</ol>

<em>,                                                </em>(1)

where and <em>λ &gt; </em>0. Compute the gradient respect to <em>w<sub>i</sub></em>, i.e. .

<ol>

 <li>Show that the problem in Eq. (1) has a unique solution.</li>

</ol>

<h1>3           Decision Trees</h1>

<ol>

 <li>Suppose you want to grow a decision tree to predict the <em>accident rate </em>based on the following accident data which provides the rate of accidents in 100 observations. Which predictor variable (weather or traffic) will you choose to split in the first step to maximize the information gain?</li>

</ol>

<table width="371">

 <tbody>

  <tr>

   <td width="73">Weather</td>

   <td width="45">Traffic</td>

   <td width="99">Accident Rate</td>

   <td width="154">Number of observations</td>

  </tr>

  <tr>

   <td width="73">Sunny</td>

   <td width="45">Heavy</td>

   <td width="99">High</td>

   <td width="154">23</td>

  </tr>

  <tr>

   <td width="73">Sunny</td>

   <td width="45">Light</td>

   <td width="99">Low</td>

   <td width="154">5</td>

  </tr>

  <tr>

   <td width="73">Rainy</td>

   <td width="45">Heavy</td>

   <td width="99">High</td>

   <td width="154">50</td>

  </tr>

  <tr>

   <td width="73">Rainy</td>

   <td width="45">Light</td>

   <td width="99">Low</td>

   <td width="154">22</td>

  </tr>

 </tbody>

</table>

<ol>

 <li>Suppose in another dataset, two students experiment with decision trees. The first student runs the decision tree learning algorithm on the raw data and obtains a tree <em>T</em><sub>1</sub>. The second student, normalizes the data by subtracting the mean and dividing by the variance of the features. Then, he runs the same decision tree algorithm with the same parameters and obtains a tree <em>T</em><sub>2</sub>. How algorithm. As discussed in ESL, Section 9.2.3, the most common splitting criteria are the <em>Gini index </em>and <em>Cross-entropy</em>. Both of these can be viewed as convex surrogates for the misclassification error. Prove that, for any discrete probability distribution <em>p </em>with <em>K </em>classes, the value of the Gini index is less than or equal to the corresponding value of the cross-entropy. This implies that the Gini index more closely approximates the misclassification error.</li>

</ol>

<em>Definitions</em>: For a <em>K</em>-valued discrete random variable with probability mass function <em>p<sub>i</sub>,i </em>= 1<em>,…,K </em>the Gini index is defined as: ) and the cross-entropy is defined as .

<h1>4           Comparing Classifiers in MATLAB/Octave</h1>

In this problem, you will work with the same dataset as in <a href="http://web.cs.ucla.edu/~ameet/teaching/fall15/cs260/hw/hw1.pdf">HW1</a><a href="http://web.cs.ucla.edu/~ameet/teaching/fall15/cs260/hw/hw1.pdf">,</a> and compare the performance of various classification algorithms. Starting with the one-hot-encoded version of the data that you generated in Question 4a in HW1, perform the following steps:

<ol>

 <li>Fill in the function naive bayes in the naive m file. In particular, implement the <a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_naive_Bayes">Bernoulli </a><a href="https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Bernoulli_naive_Bayes">Naive Bayes</a> model from scratch (this will first require you to compute the MLE estimates). The inputs of this function are training data and new data (either validation or testing data). The function needs to output the accuracy on both training and new data (either validation or testing). Note that some feature values might exist in the validation/testing data that do not exist in the training data. In that case, please set the probability of that feature value to a small value, for example, 0.1. Note: You should NOT use any related Matlab toolbox functions, e.g., NaiveBayes.fit to implement Naive Bayes.</li>

 <li>Compare the four algorithms (<em>k</em>NN, Naive Bayes, Decision Tree, and Logistic Regression) on the provided dataset. For each algorithm, report accuracies as detailed below, and describe the relative performance of these algorithms in a few sentences.</li>

</ol>

<em>k</em><strong>NN: </strong>Report results from HW1.

<strong>Decision Tree: </strong>Train decision trees using the function ClassificationTree.fit or fitctree in Matlab. Report the training, validation and test accuracy for different split criterions (<em>Gini index </em>and <em>cross-entropy </em>using the SplitCriterion attribute) and different settings for the minimum size of leaf nodes to 1<em>,</em>2<em>,</em>··· <em>,</em>10 (using the MinLeaf attribute). Thus, in total you will report the results for 2 × 10 = 20 different cases. When training decision trees, turn off pruning using the Prune attribute.

<strong>Naive Bayes: </strong>Report the training, validation and test accuracy.

<strong>Logistic Regression: </strong>Train multi-class logistic regression using the function mnrfit in Matlab. Report the training, validation and test accuracy.