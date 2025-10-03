
1

Automatic Zoom
9314 IEEE TRANSACTIONS ON INDUSTRIAL INFORMATICS, VOL. 19, NO. 9, SEPTEMBER 2023
Graph Convolution Neural Network Based
End-to-End Channel Selection and
Classification for Motor Imagery
Brain–Computer Interfaces
Biao Sun , Senior Member, IEEE, Zhengkun Liu , Zexu Wu  , Chaoxu Mu , Senior Member, IEEE,
and Ting Li
Abstract—Classification of electroencephalogram-based
motor imagery (MI-EEG) tasks is crucial in brain–computer
interface (BCI). EEG signals require a large number of chan-
nels in the acquisition process, which hinders its applica-
tion in practice. How to select the optimal channel subset
without a serious impact on the classification performance
is an urgent problem to be solved in the field of BCIs.
This article proposes an end-to-end deep learning frame-
work, called EEG channel active inference neural network
(EEG-ARNN), which is based on graph convolutional neural
networks (GCN) to fully exploit the correlation of signals
in the temporal and spatial domains. Two channel selec-
tion methods, i.e., edge-selection (ES) and aggregation-
selection (AS), are proposed to select a specified number
of optimal channels automatically. Two publicly available
BCI Competition IV 2a (BCICIV 2a) dataset and PhysioNet
dataset and a self-collected dataset (TJU dataset) are used
to evaluate the performance of the proposed method. Ex-
perimental results reveal that the proposed method outper-
forms state-of-the-art methods in terms of both classifica-
tion accuracy and robustness. Using only a small number of
channels, we obtain a classification performance similar to
that of using all channels. Finally, the association between
selected channels and activated brain areas is analyzed,
Manuscript received 30 May 2022; revised 30 September 2022; ac-
cepted 26 November 2022. Date of publication 8 December 2022;
date of current version 24 July 2023. This work was supported by the
National Natural Science Foundation of China under Grant 61971303
and Grant 81971660, in part by the Chinese Academy of Medical
Science Health Innovation Projectunder Grant 2021-I2M-042, Grant
2021-I2M-058, Grant 2022-I2M-C&T-A-005, and Grant 2022-I2M-C&T-
B-012, and in part by the Tianjin Outstanding Youth Fund under Grant
20JCJQIC00230. Paper no. TII-22-2312. (Corresponding author: Ting
Li.)
This work involved human subjects or animals in its research. Ap-
proval of all ethical and experimental procedures and protocols was
granted by China Rehabilitation Research Center Ethics Committee
under Application No. CRRC-IEC-RF-SC-005-01.
Biao Sun, Zhengkun Liu, Zexu Wu, and Chaoxu Mu are
with the School of Electrical and Information Engineering, Tian-
jin University, Tianjin 300072, China (e-mail: sunbiao@tju.edu.cn;
zliu8306@gmail.com; wuzexuxuexi@163.com; cxmu@tju.edu.cn).
Ting Li is with the Institute of Biomedical Engineering, Chinese
Academy of Medical Sciences & Peking Union Medical College, Tianjin
300192, China (e-mail: t.li619@foxmail.com).
Color versions of one or more figures in this article are available at
https://doi.org/10.1109/TII.2022.3227736.
Digital Object Identifier 10.1109/TII.2022.3227736
which is important to reveal the working state of brain
during MI.
Index Terms—Brain computer interface (BCI), channel
selection, graph convolutional network (GCN), motor im-
agery (MI).
I. INTRODUCTION
BRAIN–COMPUTER interface (BCI) systems that capture
sensory-motor rhythms and event-related potentials from
the central nervous system and convert them to artificial out-
puts have shown great value in medical rehabilitation, enter-
tainment, learning, and military applications [1], [2], [3], [4].
Motor imagery (MI) can evoke SMR, which shares common
neurophysiological dynamics and sensorimotor areas with the
corresponding explicit motor execution (ME), but does not
produce real motor actions [5], [6]. As a functionally equivalent
counterpart to ME, MI is more convenient for BCI users with
some degree of motor impairment who cannot perform overt
ME tasks, making it important to study BCI. However, MI still
faces two major challenges. First, improving the performance of
MI-based classification poses a huge challenge for BCI design
and development. Second, existing algorithms usually require a
large number of channels to achieve good classification perfor-
mance, which limits the practicality of BCI systems and their
ability to be translated into the clinic.
Because of the nonstationary, time-varying, and multichan-
nels of EEG signals, traditional machine learning methods such
as Bayesian classifier [7] and support vector machine (SVM)
have limitations in achieving high classification performance.
Recently, deep artificial neural networks, loosely inspired by
biological neural networks, have shown a remarkable perfor-
mance in EEG signal classification. An et al. [8] proposed to use
multiple deep belief nets as weak classifiers and then combined
them into a stronger classifier based on the Ada-boost algorithm,
achieving a 4–6% performance improvement compared to the
SVM algorithm. A framework combining conventional neural
network(CNN) andautoencoder was proposedbyTabar et al. [9]
to classify feature which was transformed by short time distance
Fourier transform (STFT) with more significant results. The
lately proposed EEGNet [10] employed a novel scheme that
This work is licensed under a Creative Commons Attribution 4.0 License. For more information, see http://creativecommons.org/licenses/by/4.0/
SUN et al.: GRAPH CONVOLUTION NEURAL NETWORK BASED END-TO-END CHANNEL SELECTION AND CLASSIFICATION 9315
combined classification and feature extraction in one network,
and achieved relatively good results in several BCI paradigms.
Sun et al. [11], [12] added an attention mechanism to a CNN
designed to give different attention to different channels of EEG
data, achieving state-of-the-art results in current BCI applica-
tions. Although CNN models have achieved good results for MI
classification, it is worth noting that traditional CNN are better at
processing local features of signals such as speech, video, and
images, where the signals are constantly changing [13]. CNN
approaches may be less suitable for EEG signals, as EEG signals
are discrete and noncontinuous in the spatial domain.
Recent work has shown that graph neural network (GNN)
can serve as valuable models for EEG signal classification.
GNN is a novel network that use the graph theory to process
data in the graph domain, and has shown great potential for
non-Euclidean spatial domains such as image classification [14],
channel classification [15], and traffic prediction [16]. Cheb-
Net [14] was proposed to speed up the graph convolution
operation while ensuring the performance by parameterizing
the graph convolution using the Chebyshev polynomials. Based
on ChebNet, Kipf et al. [17] proposed the graph convolutional
network (GCN) by combining CNN with spectral theory. GCN is
not only better than ChebNet in terms of performance, but also
highly scalable [15]. Compared with CNN models, GCN has
the advantage in handling discriminative feature extraction of
signals [18], and more importantly, GCN offers a way to explore
the intrinsic relationships between different channels of EEG
signals. GCN has been widely used in brain signal processing
and its effectiveness has been proved. Some current methods
based on GCN made some innovations in the adjacency matrix.
Zhang et al. [19] used prior knowledge to transform the 2-D
or 3-D spatial positions of electrodes into adjacency matrix. Li
et al. [20] used mutual information to construct the adjacency
matrix. Du et al. [21] used spatial distance matrix and relational
communication matrix to initialize the adjacency matrix. How-
ever, most of the existing work has focused on the design of
adjacency matrices to improve the decoding accuracy, which
often requires manual design or requires a priori knowledge.
The use of dense electrodes for EEG recordings increases the
burden on the subjects, it is becoming increasingly evident that
novel channel selection approaches need to be explored [22].
The purpose of channel selection is to select the channels that
are most critical to classification, thereby reducing the computa-
tional complexity of the BCI system, speeding up data process-
ing, and reducing the adverse effects of irrelevant EEG channels
on classification performance. The activity of brain areas still
varies from subject to subject in the same MI task despite the
maturity of brain region delineation. Therefore, the selection of
EEG channels that are appropriate for a particular subject on
an individual basis is essential for the practical application of
MI-BCI. There have been some studies on channel selection,
including filters, wrappers, and embedded methods [23], [24],
[25]. Among these methods, the common spatial pattern (CSP)
algorithm and its variants [26], [27], [28] have received much
attention for their simplicity and efficiency. Meng et al. [29]
measured channel weight coefficients to select channels via CSP,
whose computational efficiency and accuracy cannot be satisfied
at the same time. In order to solve the channel selection problem,
Yong et al. [30] used 1 parametric regularization to enable
sparse space filters. It transforms the optimization problem into a
quadratically constrained quadratic programming problem. This
method is more accurate, but the calculation cost is high. Based
on the hypothesis that the channels related to MI should contain
common information, a correlation-based channel selection is
proposed by Jing et al. [31]. Aiming to improving classification
performance of MI-based BCI, they also used regularized CSP
to extract effective features. As a result, the highly correlated
channels were selected and achieve promising improvement.
Zhang et al. [11] proposed to use deep neural networks for
channel selection, which automatically selects channels with
higher weights by optimizing squeeze and excitation blocks with
sparse regularization. However, it does not sufficiently take into
account the spatial information between channels.
To address the above issues, this article proposes a EEG chan-
nel active inference neural network (EEG-ARNN), which not
only outperforms the state-of-the-art (SOTA) methods in terms
of accuracy and robustness, but also enables channel selection
for specific subjects. The main contributions are as follows:
1) An end-to-end EEG-ARNN method for MI classification,
which consists of temporal feature extraction module
(TFEM) and channel active reasoning module (CARM),
is proposed. The TFEM is used to extract temporal fea-
tures of EEG signals. The CARM, which is based on
GCN, eliminates the need to construct an artificial adja-
cency matrix and can continuously modify the connec-
tivity between different channels in the subject-specifical
situation.
2) Two channel selection methods, termed as edge-selection
(ES) and aggregation-selection (AS), are proposed to
choose optimal subset of channels for particular subjects.
In addition, when using selected channels to train EEG-
ARNN, classification performance close to that of full
channel data can be obtained by using only 1/6 to 1/2 of
the original data volume. This will help to simplify the
BCI setup and facilitate practical applications.
3) We explore the connection between the EEG channels
selected by ES and AS during MI and the brain regions in
which they are located, offering the possibility to further
explore the activity levels in different brain regions during
MI and paving the way for the development of practical
brain–computer interface systems.
The rest of this article is organized as follows: Section
II introduces the EEG-ARNN model, ES and AS methods.
In Section III, experimental results are presented and the
relationship between the brain regions is explored. Finally,
Section IV concludes this article.
II. METHODS
By simulation of human brain activation with GCN and
extracting the EEG features of temporal domain with CNN, a
novel MI-EEG classification framework is built in this work.
As shown in Fig. 1, EEG-ARNN mainly consists of two parts:
the CARM based on CNN and the TFEM based on GCN. In
this section, CARM, TFEM, and the whole framework detail
are described. After that, the CARM-based ES and AS methods
are described in detail.
