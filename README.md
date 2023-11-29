# GPT from Scratch
This repository contains a GPT-like model based on the [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) tutorial video.


# Environment Configuration
Ensure the [tatooine](https://github.com/atkinssamuel/tatooine) repository exists a directory above this one. 

```
├── gpt-from-scratch/
│   └── ...
└── tatooine/
    └── ...
```

Then, create a Python environment and install the dependencies in the `requirements.txt` file:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

# Model Explanation
The model is a decoder-only model. It replicates many of the components in the [Attention Is All You Need](https://1drv.ms/b/s!Aq54YqVxo3iF5THdwn7HwvjWkFrY?e=Pggt31) paper. The image below illustrates the model architecture:

![Architecture](images/architecture.png)

The input data is tokenized using a character-by-character tokenizer. Then, the tokenized input data is fed into an embedding layer which creates a token embedding. This token embedding is combined with the positional embedding (which encodes the indices of the tokens in a batch into an embedding space). This is then fed into N sequential multi-head attention + feedforward blocks. The output is then passed into a layer normalization layer and a final linear layer to generate a token prediction. 

The figure below illustrates the structure of a single multi-head attention + feedfoward block:

![Block](images/block.png)

As shown in the figure above, the input is pre-layer layer normalized and then fed into multiple stacked attention components. The output of each of these attention components is concatenated. Then, it is passed through a linear layer and a dropout layer. The resulting output is then combined with the original input through a residual connection. A similar process occurs in which the input is pre-layer normalized, passed through a square feed-forward component, and then combined again with the input through a residual connection. 

Breaking down the model further, the image below illustrates the structure of an individual attention block.

![Attention](images/attention.png)

In an attention block, the input is split three ways and passed into three linear layers. The outputs of these three linear layers are the key, query, and value vectors (K, Q, and V). These vectors are passed into the scaled dot-product attention formula and dropout is applied to the weight matrix, $QK^T$. As mentioned before, there are multiple attention "heads". The outputs of these heads are concatenated and fed into the next phase of the model.

The image below details the square feed-forward block.

![Square Feedforward](images/square-ffwd.png)

The input is passed through a linear layer, the ReLU activation function, another linear layer, and then a dropout layer. 

# Model Parameters

- `n_updates`: the number of training updates computed during the training loop
- `batch_size`: the number of X and y data pairs to include in a batch during each training update iteration
- `n_embd`: the size of the embedding space used to embed the tokens and positions
- `n_heads`: the number of attention heads to use inside each multi-head + feedforward block
- `n_layers`: the number of multi-head attention + feedforward blocks to use
- `learning_rate`: the learning rate used to update the model parameters

# Results

After training on an NVIDIA L4 GPU for about an hour, we were able to achieve a training loss of 1.1357 and a validation loss of 1.4723. Here is what the model is able to generate:

```
JULIET:
Till defy he it prejent true, since:
I have not still'ng had me no sugar and sell:
We have consider'd 'joy, comming on, brave him brows.

ROMEO:
I hope the likeness of the Lewis, use
Sugator By, balboain to prime suffer: nobling,
It is beyond out office, not to insprison
The peteration, and for eyes can be left
Against o'er it an in't.

FRIAR LAURENCE:
A las, caught in heavy clange on your hell.

DUKE VINCENTIO:
'Tis very loyal touchether: 'tis by some veins and conquer
A qualifities that rest lawful.

Nurse:
Go tell the house, I'll joy with her purpose it:
Right well, and kneel shall be my Edward be it;
And therefore, as I see, stream that wisely maintain.
But, sir, the rank of any society;
I may besides be pursued must disdain.

LADY CAPULET:
Well have you lay the book above this remimenty. The
senative if this good of most complish in him, last
guard with pies lodge to light put me and in Vienna, his
stones house: for it is true whipt in else his lady, come to him tyrannous way,
on the princise of him. You must not hear
not pass'd help to Matter Nelphamage, hear me
speak and ear it, and being in any company.
Are you so fare alive?

TYBALT:
Stay, stir, stand a sovereign:
You had enforce the law world only lay moved: but I am weary
preparant hourse, that hath you maked?

TYBALT:
Yes, that unaturn'd canstitude to your country prove
Doth alre and then your first effect the dixth;
For I will be here tender you out of your face:
Or will I'll stand between you all; if you be fir
Your bodies is telling yet to steal thus:
You speak to safeth your honour, since to refuse
And painted with most strength another.

AUTOLYCUS:
And I am hunt with your love,
If possessible to your honour in harm, but he,
It would be my native, I, if you pond you will,
I'll have yet you name to be your devisely in highness.

SICINIUS:
This are glant
Myself and many houses!

BRUTUS:
Fare you he turn'd, and you cannot meddly: procupe I
Appen these office: behind got to his faults,
This grown fold number of men's one with an human
Whose suffer'st have brought it, me; and, in atten
Of such a formal substance. You thought of this, therefore,
My abities that good for attempt. If they have keep'd thy
widow to him, drinking execution in the people wore.

BUCKINGHAM:
I'll trouble you both not, but for your he was very mine?

GLOUCESTER:
Bring him: when I please you so, marry, if
To their eyes are asleep companions. But he's
You Rome are--'! By ' farewell, my tongue,
Smile, defend mine have straw'd upon,
To make us forsake this anointed business,
And put the duke affardication a flay
And in happy and tale patientage indeeds!
I shall you know our complate prince, you must have:
I am ake to be brief, you, and know nothing
Stand that then here yu'll grace: I pray you, be gone,
If you stay to your father's very exchanded.
Now, Camillo; for the rest wakes
My patricians: know you and my suit to say
Well be the same instant ale him, his person feas is
All last that give your services.
Whose gods, how may she did, was not mean
Where once withal! He hath factious but that dragard
How she somes broke together our derage,
To murder her; if the blood duke her suith:
But comes one that affroid her was lately,
Save hath late us not; but he beeting, and let me speak
Thy tears will be thirdities as thy melty.
I shall sleep out--tormed, and leave it so--
Raze a sin, I must confess, against here here in hell.
All this jealousies did fight and cap betishes
And make my body vanish painted from this master:
Such noble people banished! break for the tree
His bloody sorrow let villary have go;
To anse thee that thus spirit on your pleasure to be
The rough as I live, as you king.

DUKE OF YORK:
O, p me!
I will not disposition.

RUTLOVETS:
What news i' the good fad.

LEONTES:
What, that I have did been, but stays, and read,
That mortaln in your leless would were maintain thus
Be reasoned your i's good hard? my mind of the worst
best unbruised.
```