import tensorflow as tf
import re

BATCH_SIZE = 50  # 128
MAX_WORDS_IN_REVIEW = 200  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector

lstm_size = 50
lstm_layers = 2
counter = 0
learning_rate = 0.0001

stop_words = set({'film','films','movie','one','movies','ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than',"a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone",
                  "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything",
                  "anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind",
                  "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt",
                  "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc",
                  "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former",
                  "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter",
                  "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is",
                  "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more",
                  "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none",
                  "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
                  "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several",
                  "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still",
                  "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein",
                  "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top",
                  "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when",
                  "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever",
                  "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    processed_review = []
    review = review.lower()
    review = re.sub(r"<br />", " ", review)         # remove <br /> pattern
    list1 = ["1","2","3","4","5","6","7","8","9","0","!","@","#","$","%","^","&","*","(",")","~","-","_","+","=","{","}","[","]","|","<",">",";",":","?"]
    l=[]
    for w in review.split():
        l.append(w)
    for w in range(len(l)):
        for i in range(len(l[w])):
            if l[w][i] in list1:
                #print(w[i])
                l[w] = " "
                break
    review = ' '.join(l)
    #print(review)
    review = re.sub('[^\w]', ' ',review)
    for word in review.split():
        # remove [, . ! ? " / \ - ( )]
        word = re.sub(r"\,", " ", word)             
        word = re.sub(r"\)", " ", word)
        word = re.sub(r"\(", " ", word)
        word = re.sub(r"\.", " ", word)
        word = re.sub(r"\!", " ", word)
        word = re.sub(r"\"", " ", word)
        word = re.sub(r"\?", " ", word)
        word = re.sub(r"\/", " ", word)
        word = re.sub(r"\\", " ", word)
        word = re.sub(r"\-", " ", word)
        for w in word.split():
            # remove ' if at end or beginning
            if w[-1] == "'":
                w = w[:-1]
            if len(w) > 0 and w[0] == "'":
                w = w[1:]
            # remove stop-words and single-letter words
            if (w not in stop_words and len(w) > 3) or w == "bad":
                processed_review.append(w)
    
    return processed_review


def lstm_cell(dropout_keep_prob):

    global counter
    counter+=1
    global lstm_size
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size, reuse=tf.get_variable_scope().reuse)
    lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob)
    #if counter == 2:
     #   lstm_size = lstm_size - 25
    return lstm

def get_accuracy_definition(preds_op, labels):
    correct_preds_op = tf.equal(tf.argmax(preds_op, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds_op, tf.float32), name="accuracy")
    return accuracy

def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name="input_data")
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, 2], name="labels")
    
    dropout_keep_prob = tf.placeholder_with_default(0.6, shape=(), name = "dropout_keep_prob")
    
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(dropout_keep_prob) for _ in range(lstm_layers)])
    
    initial_state = cell.zero_state(BATCH_SIZE, tf.float32)
    
    outputs, final_state = tf.nn.dynamic_rnn(cell, input_data, initial_state=initial_state)
    
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 2, activation_fn=tf.nn.softmax)
    
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels = labels, logits = predictions), name="loss")
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    
    Accuracy = get_accuracy_definition(predictions, labels)

    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
