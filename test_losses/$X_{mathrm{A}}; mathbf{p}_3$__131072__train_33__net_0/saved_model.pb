??$
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AssignVariableOp
resource
value"dtype"
dtypetype?
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-0-ga4dfb8d1a718??
?
m_0__to__m_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*$
shared_namem_0__to__m_1/kernel
{
'm_0__to__m_1/kernel/Read/ReadVariableOpReadVariableOpm_0__to__m_1/kernel*
_output_shapes

:>*
dtype0
z
m_0__to__m_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*"
shared_namem_0__to__m_1/bias
s
%m_0__to__m_1/bias/Read/ReadVariableOpReadVariableOpm_0__to__m_1/bias*
_output_shapes
:>*
dtype0
?
m_1__to__m_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*$
shared_namem_1__to__m_2/kernel
{
'm_1__to__m_2/kernel/Read/ReadVariableOpReadVariableOpm_1__to__m_2/kernel*
_output_shapes

:>>*
dtype0
z
m_1__to__m_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*"
shared_namem_1__to__m_2/bias
s
%m_1__to__m_2/bias/Read/ReadVariableOpReadVariableOpm_1__to__m_2/bias*
_output_shapes
:>*
dtype0
?
m_2__to__m_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*$
shared_namem_2__to__m_3/kernel
{
'm_2__to__m_3/kernel/Read/ReadVariableOpReadVariableOpm_2__to__m_3/kernel*
_output_shapes

:>>*
dtype0
z
m_2__to__m_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*"
shared_namem_2__to__m_3/bias
s
%m_2__to__m_3/bias/Read/ReadVariableOpReadVariableOpm_2__to__m_3/bias*
_output_shapes
:>*
dtype0
?
m_3__to__m_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*$
shared_namem_3__to__m_4/kernel
{
'm_3__to__m_4/kernel/Read/ReadVariableOpReadVariableOpm_3__to__m_4/kernel*
_output_shapes

:>>*
dtype0
z
m_3__to__m_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*"
shared_namem_3__to__m_4/bias
s
%m_3__to__m_4/bias/Read/ReadVariableOpReadVariableOpm_3__to__m_4/bias*
_output_shapes
:>*
dtype0
?
m_4__to__m_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*$
shared_namem_4__to__m_5/kernel
{
'm_4__to__m_5/kernel/Read/ReadVariableOpReadVariableOpm_4__to__m_5/kernel*
_output_shapes

:>>*
dtype0
z
m_4__to__m_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*"
shared_namem_4__to__m_5/bias
s
%m_4__to__m_5/bias/Read/ReadVariableOpReadVariableOpm_4__to__m_5/bias*
_output_shapes
:>*
dtype0
?
m_5__to__m_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*$
shared_namem_5__to__m_6/kernel
{
'm_5__to__m_6/kernel/Read/ReadVariableOpReadVariableOpm_5__to__m_6/kernel*
_output_shapes

:>>*
dtype0
z
m_5__to__m_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*"
shared_namem_5__to__m_6/bias
s
%m_5__to__m_6/bias/Read/ReadVariableOpReadVariableOpm_5__to__m_6/bias*
_output_shapes
:>*
dtype0
?
m_6__to__m_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*$
shared_namem_6__to__m_7/kernel
{
'm_6__to__m_7/kernel/Read/ReadVariableOpReadVariableOpm_6__to__m_7/kernel*
_output_shapes

:>>*
dtype0
z
m_6__to__m_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*"
shared_namem_6__to__m_7/bias
s
%m_6__to__m_7/bias/Read/ReadVariableOpReadVariableOpm_6__to__m_7/bias*
_output_shapes
:>*
dtype0
?
m_7__to__m_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*$
shared_namem_7__to__m_8/kernel
{
'm_7__to__m_8/kernel/Read/ReadVariableOpReadVariableOpm_7__to__m_8/kernel*
_output_shapes

:>>*
dtype0
z
m_7__to__m_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*"
shared_namem_7__to__m_8/bias
s
%m_7__to__m_8/bias/Read/ReadVariableOpReadVariableOpm_7__to__m_8/bias*
_output_shapes
:>*
dtype0
?
m_8__to__m_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*$
shared_namem_8__to__m_9/kernel
{
'm_8__to__m_9/kernel/Read/ReadVariableOpReadVariableOpm_8__to__m_9/kernel*
_output_shapes

:>>*
dtype0
z
m_8__to__m_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*"
shared_namem_8__to__m_9/bias
s
%m_8__to__m_9/bias/Read/ReadVariableOpReadVariableOpm_8__to__m_9/bias*
_output_shapes
:>*
dtype0
?
m_9__to__m_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*%
shared_namem_9__to__m_10/kernel
}
(m_9__to__m_10/kernel/Read/ReadVariableOpReadVariableOpm_9__to__m_10/kernel*
_output_shapes

:>>*
dtype0
|
m_9__to__m_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*#
shared_namem_9__to__m_10/bias
u
&m_9__to__m_10/bias/Read/ReadVariableOpReadVariableOpm_9__to__m_10/bias*
_output_shapes
:>*
dtype0
?
m_10__to__m_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*&
shared_namem_10__to__m_11/kernel

)m_10__to__m_11/kernel/Read/ReadVariableOpReadVariableOpm_10__to__m_11/kernel*
_output_shapes

:>>*
dtype0
~
m_10__to__m_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*$
shared_namem_10__to__m_11/bias
w
'm_10__to__m_11/bias/Read/ReadVariableOpReadVariableOpm_10__to__m_11/bias*
_output_shapes
:>*
dtype0
?
m_11__to__m_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*&
shared_namem_11__to__m_12/kernel

)m_11__to__m_12/kernel/Read/ReadVariableOpReadVariableOpm_11__to__m_12/kernel*
_output_shapes

:>>*
dtype0
~
m_11__to__m_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*$
shared_namem_11__to__m_12/bias
w
'm_11__to__m_12/bias/Read/ReadVariableOpReadVariableOpm_11__to__m_12/bias*
_output_shapes
:>*
dtype0
?
m_12__to__m_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*&
shared_namem_12__to__m_13/kernel

)m_12__to__m_13/kernel/Read/ReadVariableOpReadVariableOpm_12__to__m_13/kernel*
_output_shapes

:>>*
dtype0
~
m_12__to__m_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*$
shared_namem_12__to__m_13/bias
w
'm_12__to__m_13/bias/Read/ReadVariableOpReadVariableOpm_12__to__m_13/bias*
_output_shapes
:>*
dtype0
?
m_13__to__m_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*&
shared_namem_13__to__m_14/kernel

)m_13__to__m_14/kernel/Read/ReadVariableOpReadVariableOpm_13__to__m_14/kernel*
_output_shapes

:>>*
dtype0
~
m_13__to__m_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*$
shared_namem_13__to__m_14/bias
w
'm_13__to__m_14/bias/Read/ReadVariableOpReadVariableOpm_13__to__m_14/bias*
_output_shapes
:>*
dtype0
?
m_14__to__m_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*&
shared_namem_14__to__m_15/kernel

)m_14__to__m_15/kernel/Read/ReadVariableOpReadVariableOpm_14__to__m_15/kernel*
_output_shapes

:>>*
dtype0
~
m_14__to__m_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*$
shared_namem_14__to__m_15/bias
w
'm_14__to__m_15/bias/Read/ReadVariableOpReadVariableOpm_14__to__m_15/bias*
_output_shapes
:>*
dtype0
?
m_15__to__m_16/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*&
shared_namem_15__to__m_16/kernel

)m_15__to__m_16/kernel/Read/ReadVariableOpReadVariableOpm_15__to__m_16/kernel*
_output_shapes

:>*
dtype0
~
m_15__to__m_16/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namem_15__to__m_16/bias
w
'm_15__to__m_16/bias/Read/ReadVariableOpReadVariableOpm_15__to__m_16/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
Adam/m_0__to__m_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*+
shared_nameAdam/m_0__to__m_1/kernel/m
?
.Adam/m_0__to__m_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_0__to__m_1/kernel/m*
_output_shapes

:>*
dtype0
?
Adam/m_0__to__m_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_0__to__m_1/bias/m
?
,Adam/m_0__to__m_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_0__to__m_1/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_1__to__m_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_1__to__m_2/kernel/m
?
.Adam/m_1__to__m_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_1__to__m_2/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_1__to__m_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_1__to__m_2/bias/m
?
,Adam/m_1__to__m_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_1__to__m_2/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_2__to__m_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_2__to__m_3/kernel/m
?
.Adam/m_2__to__m_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_2__to__m_3/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_2__to__m_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_2__to__m_3/bias/m
?
,Adam/m_2__to__m_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_2__to__m_3/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_3__to__m_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_3__to__m_4/kernel/m
?
.Adam/m_3__to__m_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_3__to__m_4/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_3__to__m_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_3__to__m_4/bias/m
?
,Adam/m_3__to__m_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_3__to__m_4/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_4__to__m_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_4__to__m_5/kernel/m
?
.Adam/m_4__to__m_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_4__to__m_5/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_4__to__m_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_4__to__m_5/bias/m
?
,Adam/m_4__to__m_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_4__to__m_5/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_5__to__m_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_5__to__m_6/kernel/m
?
.Adam/m_5__to__m_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_5__to__m_6/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_5__to__m_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_5__to__m_6/bias/m
?
,Adam/m_5__to__m_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_5__to__m_6/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_6__to__m_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_6__to__m_7/kernel/m
?
.Adam/m_6__to__m_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_6__to__m_7/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_6__to__m_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_6__to__m_7/bias/m
?
,Adam/m_6__to__m_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_6__to__m_7/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_7__to__m_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_7__to__m_8/kernel/m
?
.Adam/m_7__to__m_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_7__to__m_8/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_7__to__m_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_7__to__m_8/bias/m
?
,Adam/m_7__to__m_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_7__to__m_8/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_8__to__m_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_8__to__m_9/kernel/m
?
.Adam/m_8__to__m_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_8__to__m_9/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_8__to__m_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_8__to__m_9/bias/m
?
,Adam/m_8__to__m_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_8__to__m_9/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_9__to__m_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*,
shared_nameAdam/m_9__to__m_10/kernel/m
?
/Adam/m_9__to__m_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_9__to__m_10/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_9__to__m_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>**
shared_nameAdam/m_9__to__m_10/bias/m
?
-Adam/m_9__to__m_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_9__to__m_10/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_10__to__m_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*-
shared_nameAdam/m_10__to__m_11/kernel/m
?
0Adam/m_10__to__m_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_10__to__m_11/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_10__to__m_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_nameAdam/m_10__to__m_11/bias/m
?
.Adam/m_10__to__m_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_10__to__m_11/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_11__to__m_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*-
shared_nameAdam/m_11__to__m_12/kernel/m
?
0Adam/m_11__to__m_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_11__to__m_12/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_11__to__m_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_nameAdam/m_11__to__m_12/bias/m
?
.Adam/m_11__to__m_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_11__to__m_12/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_12__to__m_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*-
shared_nameAdam/m_12__to__m_13/kernel/m
?
0Adam/m_12__to__m_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_12__to__m_13/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_12__to__m_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_nameAdam/m_12__to__m_13/bias/m
?
.Adam/m_12__to__m_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_12__to__m_13/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_13__to__m_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*-
shared_nameAdam/m_13__to__m_14/kernel/m
?
0Adam/m_13__to__m_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_13__to__m_14/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_13__to__m_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_nameAdam/m_13__to__m_14/bias/m
?
.Adam/m_13__to__m_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_13__to__m_14/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_14__to__m_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*-
shared_nameAdam/m_14__to__m_15/kernel/m
?
0Adam/m_14__to__m_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/kernel/m*
_output_shapes

:>>*
dtype0
?
Adam/m_14__to__m_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_nameAdam/m_14__to__m_15/bias/m
?
.Adam/m_14__to__m_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/bias/m*
_output_shapes
:>*
dtype0
?
Adam/m_15__to__m_16/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*-
shared_nameAdam/m_15__to__m_16/kernel/m
?
0Adam/m_15__to__m_16/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_15__to__m_16/kernel/m*
_output_shapes

:>*
dtype0
?
Adam/m_15__to__m_16/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m_15__to__m_16/bias/m
?
.Adam/m_15__to__m_16/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_15__to__m_16/bias/m*
_output_shapes
:*
dtype0
?
Adam/m_0__to__m_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*+
shared_nameAdam/m_0__to__m_1/kernel/v
?
.Adam/m_0__to__m_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_0__to__m_1/kernel/v*
_output_shapes

:>*
dtype0
?
Adam/m_0__to__m_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_0__to__m_1/bias/v
?
,Adam/m_0__to__m_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_0__to__m_1/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_1__to__m_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_1__to__m_2/kernel/v
?
.Adam/m_1__to__m_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_1__to__m_2/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_1__to__m_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_1__to__m_2/bias/v
?
,Adam/m_1__to__m_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_1__to__m_2/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_2__to__m_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_2__to__m_3/kernel/v
?
.Adam/m_2__to__m_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_2__to__m_3/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_2__to__m_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_2__to__m_3/bias/v
?
,Adam/m_2__to__m_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_2__to__m_3/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_3__to__m_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_3__to__m_4/kernel/v
?
.Adam/m_3__to__m_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_3__to__m_4/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_3__to__m_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_3__to__m_4/bias/v
?
,Adam/m_3__to__m_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_3__to__m_4/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_4__to__m_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_4__to__m_5/kernel/v
?
.Adam/m_4__to__m_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_4__to__m_5/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_4__to__m_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_4__to__m_5/bias/v
?
,Adam/m_4__to__m_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_4__to__m_5/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_5__to__m_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_5__to__m_6/kernel/v
?
.Adam/m_5__to__m_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_5__to__m_6/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_5__to__m_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_5__to__m_6/bias/v
?
,Adam/m_5__to__m_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_5__to__m_6/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_6__to__m_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_6__to__m_7/kernel/v
?
.Adam/m_6__to__m_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_6__to__m_7/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_6__to__m_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_6__to__m_7/bias/v
?
,Adam/m_6__to__m_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_6__to__m_7/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_7__to__m_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_7__to__m_8/kernel/v
?
.Adam/m_7__to__m_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_7__to__m_8/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_7__to__m_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_7__to__m_8/bias/v
?
,Adam/m_7__to__m_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_7__to__m_8/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_8__to__m_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*+
shared_nameAdam/m_8__to__m_9/kernel/v
?
.Adam/m_8__to__m_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_8__to__m_9/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_8__to__m_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*)
shared_nameAdam/m_8__to__m_9/bias/v
?
,Adam/m_8__to__m_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_8__to__m_9/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_9__to__m_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*,
shared_nameAdam/m_9__to__m_10/kernel/v
?
/Adam/m_9__to__m_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_9__to__m_10/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_9__to__m_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>**
shared_nameAdam/m_9__to__m_10/bias/v
?
-Adam/m_9__to__m_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_9__to__m_10/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_10__to__m_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*-
shared_nameAdam/m_10__to__m_11/kernel/v
?
0Adam/m_10__to__m_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_10__to__m_11/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_10__to__m_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_nameAdam/m_10__to__m_11/bias/v
?
.Adam/m_10__to__m_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_10__to__m_11/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_11__to__m_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*-
shared_nameAdam/m_11__to__m_12/kernel/v
?
0Adam/m_11__to__m_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_11__to__m_12/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_11__to__m_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_nameAdam/m_11__to__m_12/bias/v
?
.Adam/m_11__to__m_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_11__to__m_12/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_12__to__m_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*-
shared_nameAdam/m_12__to__m_13/kernel/v
?
0Adam/m_12__to__m_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_12__to__m_13/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_12__to__m_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_nameAdam/m_12__to__m_13/bias/v
?
.Adam/m_12__to__m_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_12__to__m_13/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_13__to__m_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*-
shared_nameAdam/m_13__to__m_14/kernel/v
?
0Adam/m_13__to__m_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_13__to__m_14/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_13__to__m_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_nameAdam/m_13__to__m_14/bias/v
?
.Adam/m_13__to__m_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_13__to__m_14/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_14__to__m_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>>*-
shared_nameAdam/m_14__to__m_15/kernel/v
?
0Adam/m_14__to__m_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/kernel/v*
_output_shapes

:>>*
dtype0
?
Adam/m_14__to__m_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:>*+
shared_nameAdam/m_14__to__m_15/bias/v
?
.Adam/m_14__to__m_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/bias/v*
_output_shapes
:>*
dtype0
?
Adam/m_15__to__m_16/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:>*-
shared_nameAdam/m_15__to__m_16/kernel/v
?
0Adam/m_15__to__m_16/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_15__to__m_16/kernel/v*
_output_shapes

:>*
dtype0
?
Adam/m_15__to__m_16/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m_15__to__m_16/bias/v
?
.Adam/m_15__to__m_16/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_15__to__m_16/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*՚
valueʚBƚ B??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
layer_with_weights-10
layer-10
layer_with_weights-11
layer-11
layer_with_weights-12
layer-12
layer_with_weights-13
layer-13
layer_with_weights-14
layer-14
layer_with_weights-15
layer-15
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
h

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
h

5kernel
6bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
h

;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
h

Akernel
Bbias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
h

Gkernel
Hbias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
h

Mkernel
Nbias
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
h

Skernel
Tbias
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
h

Ykernel
Zbias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
h

_kernel
`bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
h

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
h

kkernel
lbias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
h

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
?
witer

xbeta_1

ybeta_2
	zdecay
{learning_ratem?m?m?m?#m?$m?)m?*m?/m?0m?5m?6m?;m?<m?Am?Bm?Gm?Hm?Mm?Nm?Sm?Tm?Ym?Zm?_m?`m?em?fm?km?lm?qm?rm?v?v?v?v?#v?$v?)v?*v?/v?0v?5v?6v?;v?<v?Av?Bv?Gv?Hv?Mv?Nv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?kv?lv?qv?rv?
 
?
0
1
2
3
#4
$5
)6
*7
/8
09
510
611
;12
<13
A14
B15
G16
H17
M18
N19
S20
T21
Y22
Z23
_24
`25
e26
f27
k28
l29
q30
r31
?
0
1
2
3
#4
$5
)6
*7
/8
09
510
611
;12
<13
A14
B15
G16
H17
M18
N19
S20
T21
Y22
Z23
_24
`25
e26
f27
k28
l29
q30
r31
?
regularization_losses
|layer_regularization_losses
}metrics
~non_trainable_variables

layers
	variables
trainable_variables
?layer_metrics
 
_]
VARIABLE_VALUEm_0__to__m_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_0__to__m_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
	variables
trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_1__to__m_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_1__to__m_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
 	variables
!trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_2__to__m_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_2__to__m_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

#0
$1

#0
$1
?
%regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
&	variables
'trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_3__to__m_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_3__to__m_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
?
+regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
,	variables
-trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_4__to__m_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_4__to__m_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
?
1regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
2	variables
3trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_5__to__m_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_5__to__m_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

50
61

50
61
?
7regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
8	variables
9trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_6__to__m_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_6__to__m_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

;0
<1

;0
<1
?
=regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
>	variables
?trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_7__to__m_8/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_7__to__m_8/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

A0
B1

A0
B1
?
Cregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
D	variables
Etrainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_8__to__m_9/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_8__to__m_9/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

G0
H1
?
Iregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
J	variables
Ktrainable_variables
?layer_metrics
`^
VARIABLE_VALUEm_9__to__m_10/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEm_9__to__m_10/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

M0
N1

M0
N1
?
Oregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
P	variables
Qtrainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_10__to__m_11/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_10__to__m_11/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

S0
T1

S0
T1
?
Uregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
V	variables
Wtrainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_11__to__m_12/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_11__to__m_12/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Y0
Z1

Y0
Z1
?
[regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
\	variables
]trainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_12__to__m_13/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_12__to__m_13/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

_0
`1

_0
`1
?
aregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
b	variables
ctrainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_13__to__m_14/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_13__to__m_14/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

e0
f1
?
gregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
h	variables
itrainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_14__to__m_15/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_14__to__m_15/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

k0
l1

k0
l1
?
mregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
n	variables
otrainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_15__to__m_16/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_15__to__m_16/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE
 

q0
r1

q0
r1
?
sregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
t	variables
utrainable_variables
?layer_metrics
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 

?0
 
v
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

?total

?count
?	variables
?	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
??
VARIABLE_VALUEAdam/m_0__to__m_1/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_0__to__m_1/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_1__to__m_2/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_1__to__m_2/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_2__to__m_3/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_2__to__m_3/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_3__to__m_4/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_3__to__m_4/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_4__to__m_5/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_4__to__m_5/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_5__to__m_6/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_5__to__m_6/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_6__to__m_7/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_6__to__m_7/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_7__to__m_8/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_7__to__m_8/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_8__to__m_9/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_8__to__m_9/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_9__to__m_10/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/m_9__to__m_10/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_10__to__m_11/kernel/mSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_10__to__m_11/bias/mQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_11__to__m_12/kernel/mSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_11__to__m_12/bias/mQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_12__to__m_13/kernel/mSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_12__to__m_13/bias/mQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_13__to__m_14/kernel/mSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_13__to__m_14/bias/mQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_14__to__m_15/kernel/mSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_14__to__m_15/bias/mQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_15__to__m_16/kernel/mSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_15__to__m_16/bias/mQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_0__to__m_1/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_0__to__m_1/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_1__to__m_2/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_1__to__m_2/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_2__to__m_3/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_2__to__m_3/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_3__to__m_4/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_3__to__m_4/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_4__to__m_5/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_4__to__m_5/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_5__to__m_6/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_5__to__m_6/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_6__to__m_7/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_6__to__m_7/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_7__to__m_8/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_7__to__m_8/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_8__to__m_9/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUEAdam/m_8__to__m_9/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_9__to__m_10/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/m_9__to__m_10/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_10__to__m_11/kernel/vSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_10__to__m_11/bias/vQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_11__to__m_12/kernel/vSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_11__to__m_12/bias/vQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_12__to__m_13/kernel/vSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_12__to__m_13/bias/vQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_13__to__m_14/kernel/vSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_13__to__m_14/bias/vQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_14__to__m_15/kernel/vSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_14__to__m_15/bias/vQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUEAdam/m_15__to__m_16/kernel/vSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/m_15__to__m_16/bias/vQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
"serving_default_m_0__to__m_1_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_m_0__to__m_1_inputm_0__to__m_1/kernelm_0__to__m_1/biasm_1__to__m_2/kernelm_1__to__m_2/biasm_2__to__m_3/kernelm_2__to__m_3/biasm_3__to__m_4/kernelm_3__to__m_4/biasm_4__to__m_5/kernelm_4__to__m_5/biasm_5__to__m_6/kernelm_5__to__m_6/biasm_6__to__m_7/kernelm_6__to__m_7/biasm_7__to__m_8/kernelm_7__to__m_8/biasm_8__to__m_9/kernelm_8__to__m_9/biasm_9__to__m_10/kernelm_9__to__m_10/biasm_10__to__m_11/kernelm_10__to__m_11/biasm_11__to__m_12/kernelm_11__to__m_12/biasm_12__to__m_13/kernelm_12__to__m_13/biasm_13__to__m_14/kernelm_13__to__m_14/biasm_14__to__m_15/kernelm_14__to__m_15/biasm_15__to__m_16/kernelm_15__to__m_16/bias*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_13991128
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?&
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'm_0__to__m_1/kernel/Read/ReadVariableOp%m_0__to__m_1/bias/Read/ReadVariableOp'm_1__to__m_2/kernel/Read/ReadVariableOp%m_1__to__m_2/bias/Read/ReadVariableOp'm_2__to__m_3/kernel/Read/ReadVariableOp%m_2__to__m_3/bias/Read/ReadVariableOp'm_3__to__m_4/kernel/Read/ReadVariableOp%m_3__to__m_4/bias/Read/ReadVariableOp'm_4__to__m_5/kernel/Read/ReadVariableOp%m_4__to__m_5/bias/Read/ReadVariableOp'm_5__to__m_6/kernel/Read/ReadVariableOp%m_5__to__m_6/bias/Read/ReadVariableOp'm_6__to__m_7/kernel/Read/ReadVariableOp%m_6__to__m_7/bias/Read/ReadVariableOp'm_7__to__m_8/kernel/Read/ReadVariableOp%m_7__to__m_8/bias/Read/ReadVariableOp'm_8__to__m_9/kernel/Read/ReadVariableOp%m_8__to__m_9/bias/Read/ReadVariableOp(m_9__to__m_10/kernel/Read/ReadVariableOp&m_9__to__m_10/bias/Read/ReadVariableOp)m_10__to__m_11/kernel/Read/ReadVariableOp'm_10__to__m_11/bias/Read/ReadVariableOp)m_11__to__m_12/kernel/Read/ReadVariableOp'm_11__to__m_12/bias/Read/ReadVariableOp)m_12__to__m_13/kernel/Read/ReadVariableOp'm_12__to__m_13/bias/Read/ReadVariableOp)m_13__to__m_14/kernel/Read/ReadVariableOp'm_13__to__m_14/bias/Read/ReadVariableOp)m_14__to__m_15/kernel/Read/ReadVariableOp'm_14__to__m_15/bias/Read/ReadVariableOp)m_15__to__m_16/kernel/Read/ReadVariableOp'm_15__to__m_16/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/m_0__to__m_1/kernel/m/Read/ReadVariableOp,Adam/m_0__to__m_1/bias/m/Read/ReadVariableOp.Adam/m_1__to__m_2/kernel/m/Read/ReadVariableOp,Adam/m_1__to__m_2/bias/m/Read/ReadVariableOp.Adam/m_2__to__m_3/kernel/m/Read/ReadVariableOp,Adam/m_2__to__m_3/bias/m/Read/ReadVariableOp.Adam/m_3__to__m_4/kernel/m/Read/ReadVariableOp,Adam/m_3__to__m_4/bias/m/Read/ReadVariableOp.Adam/m_4__to__m_5/kernel/m/Read/ReadVariableOp,Adam/m_4__to__m_5/bias/m/Read/ReadVariableOp.Adam/m_5__to__m_6/kernel/m/Read/ReadVariableOp,Adam/m_5__to__m_6/bias/m/Read/ReadVariableOp.Adam/m_6__to__m_7/kernel/m/Read/ReadVariableOp,Adam/m_6__to__m_7/bias/m/Read/ReadVariableOp.Adam/m_7__to__m_8/kernel/m/Read/ReadVariableOp,Adam/m_7__to__m_8/bias/m/Read/ReadVariableOp.Adam/m_8__to__m_9/kernel/m/Read/ReadVariableOp,Adam/m_8__to__m_9/bias/m/Read/ReadVariableOp/Adam/m_9__to__m_10/kernel/m/Read/ReadVariableOp-Adam/m_9__to__m_10/bias/m/Read/ReadVariableOp0Adam/m_10__to__m_11/kernel/m/Read/ReadVariableOp.Adam/m_10__to__m_11/bias/m/Read/ReadVariableOp0Adam/m_11__to__m_12/kernel/m/Read/ReadVariableOp.Adam/m_11__to__m_12/bias/m/Read/ReadVariableOp0Adam/m_12__to__m_13/kernel/m/Read/ReadVariableOp.Adam/m_12__to__m_13/bias/m/Read/ReadVariableOp0Adam/m_13__to__m_14/kernel/m/Read/ReadVariableOp.Adam/m_13__to__m_14/bias/m/Read/ReadVariableOp0Adam/m_14__to__m_15/kernel/m/Read/ReadVariableOp.Adam/m_14__to__m_15/bias/m/Read/ReadVariableOp0Adam/m_15__to__m_16/kernel/m/Read/ReadVariableOp.Adam/m_15__to__m_16/bias/m/Read/ReadVariableOp.Adam/m_0__to__m_1/kernel/v/Read/ReadVariableOp,Adam/m_0__to__m_1/bias/v/Read/ReadVariableOp.Adam/m_1__to__m_2/kernel/v/Read/ReadVariableOp,Adam/m_1__to__m_2/bias/v/Read/ReadVariableOp.Adam/m_2__to__m_3/kernel/v/Read/ReadVariableOp,Adam/m_2__to__m_3/bias/v/Read/ReadVariableOp.Adam/m_3__to__m_4/kernel/v/Read/ReadVariableOp,Adam/m_3__to__m_4/bias/v/Read/ReadVariableOp.Adam/m_4__to__m_5/kernel/v/Read/ReadVariableOp,Adam/m_4__to__m_5/bias/v/Read/ReadVariableOp.Adam/m_5__to__m_6/kernel/v/Read/ReadVariableOp,Adam/m_5__to__m_6/bias/v/Read/ReadVariableOp.Adam/m_6__to__m_7/kernel/v/Read/ReadVariableOp,Adam/m_6__to__m_7/bias/v/Read/ReadVariableOp.Adam/m_7__to__m_8/kernel/v/Read/ReadVariableOp,Adam/m_7__to__m_8/bias/v/Read/ReadVariableOp.Adam/m_8__to__m_9/kernel/v/Read/ReadVariableOp,Adam/m_8__to__m_9/bias/v/Read/ReadVariableOp/Adam/m_9__to__m_10/kernel/v/Read/ReadVariableOp-Adam/m_9__to__m_10/bias/v/Read/ReadVariableOp0Adam/m_10__to__m_11/kernel/v/Read/ReadVariableOp.Adam/m_10__to__m_11/bias/v/Read/ReadVariableOp0Adam/m_11__to__m_12/kernel/v/Read/ReadVariableOp.Adam/m_11__to__m_12/bias/v/Read/ReadVariableOp0Adam/m_12__to__m_13/kernel/v/Read/ReadVariableOp.Adam/m_12__to__m_13/bias/v/Read/ReadVariableOp0Adam/m_13__to__m_14/kernel/v/Read/ReadVariableOp.Adam/m_13__to__m_14/bias/v/Read/ReadVariableOp0Adam/m_14__to__m_15/kernel/v/Read/ReadVariableOp.Adam/m_14__to__m_15/bias/v/Read/ReadVariableOp0Adam/m_15__to__m_16/kernel/v/Read/ReadVariableOp.Adam/m_15__to__m_16/bias/v/Read/ReadVariableOpConst*t
Tinm
k2i	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_save_13992710
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamem_0__to__m_1/kernelm_0__to__m_1/biasm_1__to__m_2/kernelm_1__to__m_2/biasm_2__to__m_3/kernelm_2__to__m_3/biasm_3__to__m_4/kernelm_3__to__m_4/biasm_4__to__m_5/kernelm_4__to__m_5/biasm_5__to__m_6/kernelm_5__to__m_6/biasm_6__to__m_7/kernelm_6__to__m_7/biasm_7__to__m_8/kernelm_7__to__m_8/biasm_8__to__m_9/kernelm_8__to__m_9/biasm_9__to__m_10/kernelm_9__to__m_10/biasm_10__to__m_11/kernelm_10__to__m_11/biasm_11__to__m_12/kernelm_11__to__m_12/biasm_12__to__m_13/kernelm_12__to__m_13/biasm_13__to__m_14/kernelm_13__to__m_14/biasm_14__to__m_15/kernelm_14__to__m_15/biasm_15__to__m_16/kernelm_15__to__m_16/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/m_0__to__m_1/kernel/mAdam/m_0__to__m_1/bias/mAdam/m_1__to__m_2/kernel/mAdam/m_1__to__m_2/bias/mAdam/m_2__to__m_3/kernel/mAdam/m_2__to__m_3/bias/mAdam/m_3__to__m_4/kernel/mAdam/m_3__to__m_4/bias/mAdam/m_4__to__m_5/kernel/mAdam/m_4__to__m_5/bias/mAdam/m_5__to__m_6/kernel/mAdam/m_5__to__m_6/bias/mAdam/m_6__to__m_7/kernel/mAdam/m_6__to__m_7/bias/mAdam/m_7__to__m_8/kernel/mAdam/m_7__to__m_8/bias/mAdam/m_8__to__m_9/kernel/mAdam/m_8__to__m_9/bias/mAdam/m_9__to__m_10/kernel/mAdam/m_9__to__m_10/bias/mAdam/m_10__to__m_11/kernel/mAdam/m_10__to__m_11/bias/mAdam/m_11__to__m_12/kernel/mAdam/m_11__to__m_12/bias/mAdam/m_12__to__m_13/kernel/mAdam/m_12__to__m_13/bias/mAdam/m_13__to__m_14/kernel/mAdam/m_13__to__m_14/bias/mAdam/m_14__to__m_15/kernel/mAdam/m_14__to__m_15/bias/mAdam/m_15__to__m_16/kernel/mAdam/m_15__to__m_16/bias/mAdam/m_0__to__m_1/kernel/vAdam/m_0__to__m_1/bias/vAdam/m_1__to__m_2/kernel/vAdam/m_1__to__m_2/bias/vAdam/m_2__to__m_3/kernel/vAdam/m_2__to__m_3/bias/vAdam/m_3__to__m_4/kernel/vAdam/m_3__to__m_4/bias/vAdam/m_4__to__m_5/kernel/vAdam/m_4__to__m_5/bias/vAdam/m_5__to__m_6/kernel/vAdam/m_5__to__m_6/bias/vAdam/m_6__to__m_7/kernel/vAdam/m_6__to__m_7/bias/vAdam/m_7__to__m_8/kernel/vAdam/m_7__to__m_8/bias/vAdam/m_8__to__m_9/kernel/vAdam/m_8__to__m_9/bias/vAdam/m_9__to__m_10/kernel/vAdam/m_9__to__m_10/bias/vAdam/m_10__to__m_11/kernel/vAdam/m_10__to__m_11/bias/vAdam/m_11__to__m_12/kernel/vAdam/m_11__to__m_12/bias/vAdam/m_12__to__m_13/kernel/vAdam/m_12__to__m_13/bias/vAdam/m_13__to__m_14/kernel/vAdam/m_13__to__m_14/bias/vAdam/m_14__to__m_15/kernel/vAdam/m_14__to__m_15/bias/vAdam/m_15__to__m_16/kernel/vAdam/m_15__to__m_16/bias/v*s
Tinl
j2h*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference__traced_restore_13993029??
?
?
__inference_loss_fn_0_13992213M
;m_0__to__m_1_kernel_regularizer_abs_readvariableop_resource:>
identity??2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_0__to__m_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2%
#m_0__to__m_1/kernel/Regularizer/Abs?
%m_0__to__m_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_0__to__m_1/kernel/Regularizer/Const?
#m_0__to__m_1/kernel/Regularizer/SumSum'm_0__to__m_1/kernel/Regularizer/Abs:y:0.m_0__to__m_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/Sum?
%m_0__to__m_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_0__to__m_1/kernel/Regularizer/mul/x?
#m_0__to__m_1/kernel/Regularizer/mulMul.m_0__to__m_1/kernel/Regularizer/mul/x:output:0,m_0__to__m_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/mul?
IdentityIdentity'm_0__to__m_1/kernel/Regularizer/mul:z:03^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp
?
?
__inference_loss_fn_9_13992312N
<m_9__to__m_10_kernel_regularizer_abs_readvariableop_resource:>>
identity??3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<m_9__to__m_10_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2&
$m_9__to__m_10/kernel/Regularizer/Abs?
&m_9__to__m_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&m_9__to__m_10/kernel/Regularizer/Const?
$m_9__to__m_10/kernel/Regularizer/SumSum(m_9__to__m_10/kernel/Regularizer/Abs:y:0/m_9__to__m_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/Sum?
&m_9__to__m_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92(
&m_9__to__m_10/kernel/Regularizer/mul/x?
$m_9__to__m_10/kernel/Regularizer/mulMul/m_9__to__m_10/kernel/Regularizer/mul/x:output:0-m_9__to__m_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/mul?
IdentityIdentity(m_9__to__m_10/kernel/Regularizer/mul:z:04^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp
??
?
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13990459

inputs'
m_0__to__m_1_13990282:>#
m_0__to__m_1_13990284:>'
m_1__to__m_2_13990287:>>#
m_1__to__m_2_13990289:>'
m_2__to__m_3_13990292:>>#
m_2__to__m_3_13990294:>'
m_3__to__m_4_13990297:>>#
m_3__to__m_4_13990299:>'
m_4__to__m_5_13990302:>>#
m_4__to__m_5_13990304:>'
m_5__to__m_6_13990307:>>#
m_5__to__m_6_13990309:>'
m_6__to__m_7_13990312:>>#
m_6__to__m_7_13990314:>'
m_7__to__m_8_13990317:>>#
m_7__to__m_8_13990319:>'
m_8__to__m_9_13990322:>>#
m_8__to__m_9_13990324:>(
m_9__to__m_10_13990327:>>$
m_9__to__m_10_13990329:>)
m_10__to__m_11_13990332:>>%
m_10__to__m_11_13990334:>)
m_11__to__m_12_13990337:>>%
m_11__to__m_12_13990339:>)
m_12__to__m_13_13990342:>>%
m_12__to__m_13_13990344:>)
m_13__to__m_14_13990347:>>%
m_13__to__m_14_13990349:>)
m_14__to__m_15_13990352:>>%
m_14__to__m_15_13990354:>)
m_15__to__m_16_13990357:>%
m_15__to__m_16_13990359:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?&m_15__to__m_16/StatefulPartitionedCall?4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallinputsm_0__to__m_1_13990282m_0__to__m_1_13990284*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_139895332&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_13990287m_1__to__m_2_13990289*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_139895562&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_13990292m_2__to__m_3_13990294*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_139895792&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_13990297m_3__to__m_4_13990299*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_139896022&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_13990302m_4__to__m_5_13990304*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_139896252&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_13990307m_5__to__m_6_13990309*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_139896482&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_13990312m_6__to__m_7_13990314*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_139896712&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_13990317m_7__to__m_8_13990319*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_139896942&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_13990322m_8__to__m_9_13990324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_139897172&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_13990327m_9__to__m_10_13990329*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_139897402'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_13990332m_10__to__m_11_13990334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_139897632(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_13990337m_11__to__m_12_13990339*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_139897862(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_13990342m_12__to__m_13_13990344*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_139898092(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_13990347m_13__to__m_14_13990349*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_139898322(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_13990352m_14__to__m_15_13990354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_139898552(
&m_14__to__m_15/StatefulPartitionedCall?
&m_15__to__m_16/StatefulPartitionedCallStatefulPartitionedCall/m_14__to__m_15/StatefulPartitionedCall:output:0m_15__to__m_16_13990357m_15__to__m_16_13990359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_15__to__m_16_layer_call_and_return_conditional_losses_139898782(
&m_15__to__m_16/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_13990282*
_output_shapes

:>*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2%
#m_0__to__m_1/kernel/Regularizer/Abs?
%m_0__to__m_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_0__to__m_1/kernel/Regularizer/Const?
#m_0__to__m_1/kernel/Regularizer/SumSum'm_0__to__m_1/kernel/Regularizer/Abs:y:0.m_0__to__m_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/Sum?
%m_0__to__m_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_0__to__m_1/kernel/Regularizer/mul/x?
#m_0__to__m_1/kernel/Regularizer/mulMul.m_0__to__m_1/kernel/Regularizer/mul/x:output:0,m_0__to__m_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/mul?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_13990287*
_output_shapes

:>>*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_1__to__m_2/kernel/Regularizer/Abs?
%m_1__to__m_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_1__to__m_2/kernel/Regularizer/Const?
#m_1__to__m_2/kernel/Regularizer/SumSum'm_1__to__m_2/kernel/Regularizer/Abs:y:0.m_1__to__m_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/Sum?
%m_1__to__m_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_1__to__m_2/kernel/Regularizer/mul/x?
#m_1__to__m_2/kernel/Regularizer/mulMul.m_1__to__m_2/kernel/Regularizer/mul/x:output:0,m_1__to__m_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/mul?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_13990292*
_output_shapes

:>>*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_2__to__m_3/kernel/Regularizer/Abs?
%m_2__to__m_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_2__to__m_3/kernel/Regularizer/Const?
#m_2__to__m_3/kernel/Regularizer/SumSum'm_2__to__m_3/kernel/Regularizer/Abs:y:0.m_2__to__m_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/Sum?
%m_2__to__m_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_2__to__m_3/kernel/Regularizer/mul/x?
#m_2__to__m_3/kernel/Regularizer/mulMul.m_2__to__m_3/kernel/Regularizer/mul/x:output:0,m_2__to__m_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/mul?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_13990297*
_output_shapes

:>>*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_3__to__m_4/kernel/Regularizer/Abs?
%m_3__to__m_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_3__to__m_4/kernel/Regularizer/Const?
#m_3__to__m_4/kernel/Regularizer/SumSum'm_3__to__m_4/kernel/Regularizer/Abs:y:0.m_3__to__m_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/Sum?
%m_3__to__m_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_3__to__m_4/kernel/Regularizer/mul/x?
#m_3__to__m_4/kernel/Regularizer/mulMul.m_3__to__m_4/kernel/Regularizer/mul/x:output:0,m_3__to__m_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/mul?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_13990302*
_output_shapes

:>>*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_4__to__m_5/kernel/Regularizer/Abs?
%m_4__to__m_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_4__to__m_5/kernel/Regularizer/Const?
#m_4__to__m_5/kernel/Regularizer/SumSum'm_4__to__m_5/kernel/Regularizer/Abs:y:0.m_4__to__m_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/Sum?
%m_4__to__m_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_4__to__m_5/kernel/Regularizer/mul/x?
#m_4__to__m_5/kernel/Regularizer/mulMul.m_4__to__m_5/kernel/Regularizer/mul/x:output:0,m_4__to__m_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/mul?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_13990307*
_output_shapes

:>>*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_5__to__m_6/kernel/Regularizer/Abs?
%m_5__to__m_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_5__to__m_6/kernel/Regularizer/Const?
#m_5__to__m_6/kernel/Regularizer/SumSum'm_5__to__m_6/kernel/Regularizer/Abs:y:0.m_5__to__m_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/Sum?
%m_5__to__m_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_5__to__m_6/kernel/Regularizer/mul/x?
#m_5__to__m_6/kernel/Regularizer/mulMul.m_5__to__m_6/kernel/Regularizer/mul/x:output:0,m_5__to__m_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/mul?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_13990312*
_output_shapes

:>>*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_6__to__m_7/kernel/Regularizer/Abs?
%m_6__to__m_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_6__to__m_7/kernel/Regularizer/Const?
#m_6__to__m_7/kernel/Regularizer/SumSum'm_6__to__m_7/kernel/Regularizer/Abs:y:0.m_6__to__m_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/Sum?
%m_6__to__m_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_6__to__m_7/kernel/Regularizer/mul/x?
#m_6__to__m_7/kernel/Regularizer/mulMul.m_6__to__m_7/kernel/Regularizer/mul/x:output:0,m_6__to__m_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/mul?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_13990317*
_output_shapes

:>>*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_7__to__m_8/kernel/Regularizer/Abs?
%m_7__to__m_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_7__to__m_8/kernel/Regularizer/Const?
#m_7__to__m_8/kernel/Regularizer/SumSum'm_7__to__m_8/kernel/Regularizer/Abs:y:0.m_7__to__m_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/Sum?
%m_7__to__m_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_7__to__m_8/kernel/Regularizer/mul/x?
#m_7__to__m_8/kernel/Regularizer/mulMul.m_7__to__m_8/kernel/Regularizer/mul/x:output:0,m_7__to__m_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/mul?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_13990322*
_output_shapes

:>>*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_8__to__m_9/kernel/Regularizer/Abs?
%m_8__to__m_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_8__to__m_9/kernel/Regularizer/Const?
#m_8__to__m_9/kernel/Regularizer/SumSum'm_8__to__m_9/kernel/Regularizer/Abs:y:0.m_8__to__m_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/Sum?
%m_8__to__m_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_8__to__m_9/kernel/Regularizer/mul/x?
#m_8__to__m_9/kernel/Regularizer/mulMul.m_8__to__m_9/kernel/Regularizer/mul/x:output:0,m_8__to__m_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/mul?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_13990327*
_output_shapes

:>>*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2&
$m_9__to__m_10/kernel/Regularizer/Abs?
&m_9__to__m_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&m_9__to__m_10/kernel/Regularizer/Const?
$m_9__to__m_10/kernel/Regularizer/SumSum(m_9__to__m_10/kernel/Regularizer/Abs:y:0/m_9__to__m_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/Sum?
&m_9__to__m_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92(
&m_9__to__m_10/kernel/Regularizer/mul/x?
$m_9__to__m_10/kernel/Regularizer/mulMul/m_9__to__m_10/kernel/Regularizer/mul/x:output:0-m_9__to__m_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/mul?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_13990332*
_output_shapes

:>>*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_10__to__m_11/kernel/Regularizer/Abs?
'm_10__to__m_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_10__to__m_11/kernel/Regularizer/Const?
%m_10__to__m_11/kernel/Regularizer/SumSum)m_10__to__m_11/kernel/Regularizer/Abs:y:00m_10__to__m_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/Sum?
'm_10__to__m_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_10__to__m_11/kernel/Regularizer/mul/x?
%m_10__to__m_11/kernel/Regularizer/mulMul0m_10__to__m_11/kernel/Regularizer/mul/x:output:0.m_10__to__m_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/mul?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_13990337*
_output_shapes

:>>*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_11__to__m_12/kernel/Regularizer/Abs?
'm_11__to__m_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_11__to__m_12/kernel/Regularizer/Const?
%m_11__to__m_12/kernel/Regularizer/SumSum)m_11__to__m_12/kernel/Regularizer/Abs:y:00m_11__to__m_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/Sum?
'm_11__to__m_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_11__to__m_12/kernel/Regularizer/mul/x?
%m_11__to__m_12/kernel/Regularizer/mulMul0m_11__to__m_12/kernel/Regularizer/mul/x:output:0.m_11__to__m_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/mul?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_13990342*
_output_shapes

:>>*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_12__to__m_13/kernel/Regularizer/Abs?
'm_12__to__m_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_12__to__m_13/kernel/Regularizer/Const?
%m_12__to__m_13/kernel/Regularizer/SumSum)m_12__to__m_13/kernel/Regularizer/Abs:y:00m_12__to__m_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/Sum?
'm_12__to__m_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_12__to__m_13/kernel/Regularizer/mul/x?
%m_12__to__m_13/kernel/Regularizer/mulMul0m_12__to__m_13/kernel/Regularizer/mul/x:output:0.m_12__to__m_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/mul?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_13990347*
_output_shapes

:>>*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_13__to__m_14/kernel/Regularizer/Abs?
'm_13__to__m_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_13__to__m_14/kernel/Regularizer/Const?
%m_13__to__m_14/kernel/Regularizer/SumSum)m_13__to__m_14/kernel/Regularizer/Abs:y:00m_13__to__m_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/Sum?
'm_13__to__m_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_13__to__m_14/kernel/Regularizer/mul/x?
%m_13__to__m_14/kernel/Regularizer/mulMul0m_13__to__m_14/kernel/Regularizer/mul/x:output:0.m_13__to__m_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/mul?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_13990352*
_output_shapes

:>>*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_14__to__m_15/kernel/Regularizer/Abs?
'm_14__to__m_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_14__to__m_15/kernel/Regularizer/Const?
%m_14__to__m_15/kernel/Regularizer/SumSum)m_14__to__m_15/kernel/Regularizer/Abs:y:00m_14__to__m_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/Sum?
'm_14__to__m_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_14__to__m_15/kernel/Regularizer/mul/x?
%m_14__to__m_15/kernel/Regularizer/mulMul0m_14__to__m_15/kernel/Regularizer/mul/x:output:0.m_14__to__m_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/mul?
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_15__to__m_16_13990357*
_output_shapes

:>*
dtype026
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
%m_15__to__m_16/kernel/Regularizer/AbsAbs<m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2'
%m_15__to__m_16/kernel/Regularizer/Abs?
'm_15__to__m_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_15__to__m_16/kernel/Regularizer/Const?
%m_15__to__m_16/kernel/Regularizer/SumSum)m_15__to__m_16/kernel/Regularizer/Abs:y:00m_15__to__m_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/Sum?
'm_15__to__m_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_15__to__m_16/kernel/Regularizer/mul/x?
%m_15__to__m_16/kernel/Regularizer/mulMul0m_15__to__m_16/kernel/Regularizer/mul/x:output:0.m_15__to__m_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/mul?
IdentityIdentity/m_15__to__m_16/StatefulPartitionedCall:output:0%^m_0__to__m_1/StatefulPartitionedCall3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp'^m_10__to__m_11/StatefulPartitionedCall5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp'^m_11__to__m_12/StatefulPartitionedCall5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp'^m_12__to__m_13/StatefulPartitionedCall5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp'^m_13__to__m_14/StatefulPartitionedCall5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp'^m_14__to__m_15/StatefulPartitionedCall5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp'^m_15__to__m_16/StatefulPartitionedCall5^m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp%^m_1__to__m_2/StatefulPartitionedCall3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp%^m_2__to__m_3/StatefulPartitionedCall3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp%^m_3__to__m_4/StatefulPartitionedCall3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp%^m_4__to__m_5/StatefulPartitionedCall3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp%^m_5__to__m_6/StatefulPartitionedCall3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp%^m_6__to__m_7/StatefulPartitionedCall3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp%^m_7__to__m_8/StatefulPartitionedCall3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp%^m_8__to__m_9/StatefulPartitionedCall3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp&^m_9__to__m_10/StatefulPartitionedCall4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$m_0__to__m_1/StatefulPartitionedCall$m_0__to__m_1/StatefulPartitionedCall2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2P
&m_10__to__m_11/StatefulPartitionedCall&m_10__to__m_11/StatefulPartitionedCall2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp2P
&m_11__to__m_12/StatefulPartitionedCall&m_11__to__m_12/StatefulPartitionedCall2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp2P
&m_12__to__m_13/StatefulPartitionedCall&m_12__to__m_13/StatefulPartitionedCall2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp2P
&m_13__to__m_14/StatefulPartitionedCall&m_13__to__m_14/StatefulPartitionedCall2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp2P
&m_14__to__m_15/StatefulPartitionedCall&m_14__to__m_15/StatefulPartitionedCall2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2P
&m_15__to__m_16/StatefulPartitionedCall&m_15__to__m_16/StatefulPartitionedCall2l
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp2L
$m_1__to__m_2/StatefulPartitionedCall$m_1__to__m_2/StatefulPartitionedCall2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2L
$m_2__to__m_3/StatefulPartitionedCall$m_2__to__m_3/StatefulPartitionedCall2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2L
$m_3__to__m_4/StatefulPartitionedCall$m_3__to__m_4/StatefulPartitionedCall2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2L
$m_4__to__m_5/StatefulPartitionedCall$m_4__to__m_5/StatefulPartitionedCall2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2L
$m_5__to__m_6/StatefulPartitionedCall$m_5__to__m_6/StatefulPartitionedCall2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2L
$m_6__to__m_7/StatefulPartitionedCall$m_6__to__m_7/StatefulPartitionedCall2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2L
$m_7__to__m_8/StatefulPartitionedCall$m_7__to__m_8/StatefulPartitionedCall2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2L
$m_8__to__m_9/StatefulPartitionedCall$m_8__to__m_9/StatefulPartitionedCall2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2N
%m_9__to__m_10/StatefulPartitionedCall%m_9__to__m_10/StatefulPartitionedCall2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_13989648

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_5__to__m_6/kernel/Regularizer/Abs?
%m_5__to__m_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_5__to__m_6/kernel/Regularizer/Const?
#m_5__to__m_6/kernel/Regularizer/SumSum'm_5__to__m_6/kernel/Regularizer/Abs:y:0.m_5__to__m_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/Sum?
%m_5__to__m_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_5__to__m_6/kernel/Regularizer/mul/x?
#m_5__to__m_6/kernel/Regularizer/mulMul.m_5__to__m_6/kernel/Regularizer/mul/x:output:0,m_5__to__m_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
??
?-
!__inference__traced_save_13992710
file_prefix2
.savev2_m_0__to__m_1_kernel_read_readvariableop0
,savev2_m_0__to__m_1_bias_read_readvariableop2
.savev2_m_1__to__m_2_kernel_read_readvariableop0
,savev2_m_1__to__m_2_bias_read_readvariableop2
.savev2_m_2__to__m_3_kernel_read_readvariableop0
,savev2_m_2__to__m_3_bias_read_readvariableop2
.savev2_m_3__to__m_4_kernel_read_readvariableop0
,savev2_m_3__to__m_4_bias_read_readvariableop2
.savev2_m_4__to__m_5_kernel_read_readvariableop0
,savev2_m_4__to__m_5_bias_read_readvariableop2
.savev2_m_5__to__m_6_kernel_read_readvariableop0
,savev2_m_5__to__m_6_bias_read_readvariableop2
.savev2_m_6__to__m_7_kernel_read_readvariableop0
,savev2_m_6__to__m_7_bias_read_readvariableop2
.savev2_m_7__to__m_8_kernel_read_readvariableop0
,savev2_m_7__to__m_8_bias_read_readvariableop2
.savev2_m_8__to__m_9_kernel_read_readvariableop0
,savev2_m_8__to__m_9_bias_read_readvariableop3
/savev2_m_9__to__m_10_kernel_read_readvariableop1
-savev2_m_9__to__m_10_bias_read_readvariableop4
0savev2_m_10__to__m_11_kernel_read_readvariableop2
.savev2_m_10__to__m_11_bias_read_readvariableop4
0savev2_m_11__to__m_12_kernel_read_readvariableop2
.savev2_m_11__to__m_12_bias_read_readvariableop4
0savev2_m_12__to__m_13_kernel_read_readvariableop2
.savev2_m_12__to__m_13_bias_read_readvariableop4
0savev2_m_13__to__m_14_kernel_read_readvariableop2
.savev2_m_13__to__m_14_bias_read_readvariableop4
0savev2_m_14__to__m_15_kernel_read_readvariableop2
.savev2_m_14__to__m_15_bias_read_readvariableop4
0savev2_m_15__to__m_16_kernel_read_readvariableop2
.savev2_m_15__to__m_16_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop9
5savev2_adam_m_0__to__m_1_kernel_m_read_readvariableop7
3savev2_adam_m_0__to__m_1_bias_m_read_readvariableop9
5savev2_adam_m_1__to__m_2_kernel_m_read_readvariableop7
3savev2_adam_m_1__to__m_2_bias_m_read_readvariableop9
5savev2_adam_m_2__to__m_3_kernel_m_read_readvariableop7
3savev2_adam_m_2__to__m_3_bias_m_read_readvariableop9
5savev2_adam_m_3__to__m_4_kernel_m_read_readvariableop7
3savev2_adam_m_3__to__m_4_bias_m_read_readvariableop9
5savev2_adam_m_4__to__m_5_kernel_m_read_readvariableop7
3savev2_adam_m_4__to__m_5_bias_m_read_readvariableop9
5savev2_adam_m_5__to__m_6_kernel_m_read_readvariableop7
3savev2_adam_m_5__to__m_6_bias_m_read_readvariableop9
5savev2_adam_m_6__to__m_7_kernel_m_read_readvariableop7
3savev2_adam_m_6__to__m_7_bias_m_read_readvariableop9
5savev2_adam_m_7__to__m_8_kernel_m_read_readvariableop7
3savev2_adam_m_7__to__m_8_bias_m_read_readvariableop9
5savev2_adam_m_8__to__m_9_kernel_m_read_readvariableop7
3savev2_adam_m_8__to__m_9_bias_m_read_readvariableop:
6savev2_adam_m_9__to__m_10_kernel_m_read_readvariableop8
4savev2_adam_m_9__to__m_10_bias_m_read_readvariableop;
7savev2_adam_m_10__to__m_11_kernel_m_read_readvariableop9
5savev2_adam_m_10__to__m_11_bias_m_read_readvariableop;
7savev2_adam_m_11__to__m_12_kernel_m_read_readvariableop9
5savev2_adam_m_11__to__m_12_bias_m_read_readvariableop;
7savev2_adam_m_12__to__m_13_kernel_m_read_readvariableop9
5savev2_adam_m_12__to__m_13_bias_m_read_readvariableop;
7savev2_adam_m_13__to__m_14_kernel_m_read_readvariableop9
5savev2_adam_m_13__to__m_14_bias_m_read_readvariableop;
7savev2_adam_m_14__to__m_15_kernel_m_read_readvariableop9
5savev2_adam_m_14__to__m_15_bias_m_read_readvariableop;
7savev2_adam_m_15__to__m_16_kernel_m_read_readvariableop9
5savev2_adam_m_15__to__m_16_bias_m_read_readvariableop9
5savev2_adam_m_0__to__m_1_kernel_v_read_readvariableop7
3savev2_adam_m_0__to__m_1_bias_v_read_readvariableop9
5savev2_adam_m_1__to__m_2_kernel_v_read_readvariableop7
3savev2_adam_m_1__to__m_2_bias_v_read_readvariableop9
5savev2_adam_m_2__to__m_3_kernel_v_read_readvariableop7
3savev2_adam_m_2__to__m_3_bias_v_read_readvariableop9
5savev2_adam_m_3__to__m_4_kernel_v_read_readvariableop7
3savev2_adam_m_3__to__m_4_bias_v_read_readvariableop9
5savev2_adam_m_4__to__m_5_kernel_v_read_readvariableop7
3savev2_adam_m_4__to__m_5_bias_v_read_readvariableop9
5savev2_adam_m_5__to__m_6_kernel_v_read_readvariableop7
3savev2_adam_m_5__to__m_6_bias_v_read_readvariableop9
5savev2_adam_m_6__to__m_7_kernel_v_read_readvariableop7
3savev2_adam_m_6__to__m_7_bias_v_read_readvariableop9
5savev2_adam_m_7__to__m_8_kernel_v_read_readvariableop7
3savev2_adam_m_7__to__m_8_bias_v_read_readvariableop9
5savev2_adam_m_8__to__m_9_kernel_v_read_readvariableop7
3savev2_adam_m_8__to__m_9_bias_v_read_readvariableop:
6savev2_adam_m_9__to__m_10_kernel_v_read_readvariableop8
4savev2_adam_m_9__to__m_10_bias_v_read_readvariableop;
7savev2_adam_m_10__to__m_11_kernel_v_read_readvariableop9
5savev2_adam_m_10__to__m_11_bias_v_read_readvariableop;
7savev2_adam_m_11__to__m_12_kernel_v_read_readvariableop9
5savev2_adam_m_11__to__m_12_bias_v_read_readvariableop;
7savev2_adam_m_12__to__m_13_kernel_v_read_readvariableop9
5savev2_adam_m_12__to__m_13_bias_v_read_readvariableop;
7savev2_adam_m_13__to__m_14_kernel_v_read_readvariableop9
5savev2_adam_m_13__to__m_14_bias_v_read_readvariableop;
7savev2_adam_m_14__to__m_15_kernel_v_read_readvariableop9
5savev2_adam_m_14__to__m_15_bias_v_read_readvariableop;
7savev2_adam_m_15__to__m_16_kernel_v_read_readvariableop9
5savev2_adam_m_15__to__m_16_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?;
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*?:
value?:B?:hB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*?
value?B?hB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?+
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_m_0__to__m_1_kernel_read_readvariableop,savev2_m_0__to__m_1_bias_read_readvariableop.savev2_m_1__to__m_2_kernel_read_readvariableop,savev2_m_1__to__m_2_bias_read_readvariableop.savev2_m_2__to__m_3_kernel_read_readvariableop,savev2_m_2__to__m_3_bias_read_readvariableop.savev2_m_3__to__m_4_kernel_read_readvariableop,savev2_m_3__to__m_4_bias_read_readvariableop.savev2_m_4__to__m_5_kernel_read_readvariableop,savev2_m_4__to__m_5_bias_read_readvariableop.savev2_m_5__to__m_6_kernel_read_readvariableop,savev2_m_5__to__m_6_bias_read_readvariableop.savev2_m_6__to__m_7_kernel_read_readvariableop,savev2_m_6__to__m_7_bias_read_readvariableop.savev2_m_7__to__m_8_kernel_read_readvariableop,savev2_m_7__to__m_8_bias_read_readvariableop.savev2_m_8__to__m_9_kernel_read_readvariableop,savev2_m_8__to__m_9_bias_read_readvariableop/savev2_m_9__to__m_10_kernel_read_readvariableop-savev2_m_9__to__m_10_bias_read_readvariableop0savev2_m_10__to__m_11_kernel_read_readvariableop.savev2_m_10__to__m_11_bias_read_readvariableop0savev2_m_11__to__m_12_kernel_read_readvariableop.savev2_m_11__to__m_12_bias_read_readvariableop0savev2_m_12__to__m_13_kernel_read_readvariableop.savev2_m_12__to__m_13_bias_read_readvariableop0savev2_m_13__to__m_14_kernel_read_readvariableop.savev2_m_13__to__m_14_bias_read_readvariableop0savev2_m_14__to__m_15_kernel_read_readvariableop.savev2_m_14__to__m_15_bias_read_readvariableop0savev2_m_15__to__m_16_kernel_read_readvariableop.savev2_m_15__to__m_16_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_m_0__to__m_1_kernel_m_read_readvariableop3savev2_adam_m_0__to__m_1_bias_m_read_readvariableop5savev2_adam_m_1__to__m_2_kernel_m_read_readvariableop3savev2_adam_m_1__to__m_2_bias_m_read_readvariableop5savev2_adam_m_2__to__m_3_kernel_m_read_readvariableop3savev2_adam_m_2__to__m_3_bias_m_read_readvariableop5savev2_adam_m_3__to__m_4_kernel_m_read_readvariableop3savev2_adam_m_3__to__m_4_bias_m_read_readvariableop5savev2_adam_m_4__to__m_5_kernel_m_read_readvariableop3savev2_adam_m_4__to__m_5_bias_m_read_readvariableop5savev2_adam_m_5__to__m_6_kernel_m_read_readvariableop3savev2_adam_m_5__to__m_6_bias_m_read_readvariableop5savev2_adam_m_6__to__m_7_kernel_m_read_readvariableop3savev2_adam_m_6__to__m_7_bias_m_read_readvariableop5savev2_adam_m_7__to__m_8_kernel_m_read_readvariableop3savev2_adam_m_7__to__m_8_bias_m_read_readvariableop5savev2_adam_m_8__to__m_9_kernel_m_read_readvariableop3savev2_adam_m_8__to__m_9_bias_m_read_readvariableop6savev2_adam_m_9__to__m_10_kernel_m_read_readvariableop4savev2_adam_m_9__to__m_10_bias_m_read_readvariableop7savev2_adam_m_10__to__m_11_kernel_m_read_readvariableop5savev2_adam_m_10__to__m_11_bias_m_read_readvariableop7savev2_adam_m_11__to__m_12_kernel_m_read_readvariableop5savev2_adam_m_11__to__m_12_bias_m_read_readvariableop7savev2_adam_m_12__to__m_13_kernel_m_read_readvariableop5savev2_adam_m_12__to__m_13_bias_m_read_readvariableop7savev2_adam_m_13__to__m_14_kernel_m_read_readvariableop5savev2_adam_m_13__to__m_14_bias_m_read_readvariableop7savev2_adam_m_14__to__m_15_kernel_m_read_readvariableop5savev2_adam_m_14__to__m_15_bias_m_read_readvariableop7savev2_adam_m_15__to__m_16_kernel_m_read_readvariableop5savev2_adam_m_15__to__m_16_bias_m_read_readvariableop5savev2_adam_m_0__to__m_1_kernel_v_read_readvariableop3savev2_adam_m_0__to__m_1_bias_v_read_readvariableop5savev2_adam_m_1__to__m_2_kernel_v_read_readvariableop3savev2_adam_m_1__to__m_2_bias_v_read_readvariableop5savev2_adam_m_2__to__m_3_kernel_v_read_readvariableop3savev2_adam_m_2__to__m_3_bias_v_read_readvariableop5savev2_adam_m_3__to__m_4_kernel_v_read_readvariableop3savev2_adam_m_3__to__m_4_bias_v_read_readvariableop5savev2_adam_m_4__to__m_5_kernel_v_read_readvariableop3savev2_adam_m_4__to__m_5_bias_v_read_readvariableop5savev2_adam_m_5__to__m_6_kernel_v_read_readvariableop3savev2_adam_m_5__to__m_6_bias_v_read_readvariableop5savev2_adam_m_6__to__m_7_kernel_v_read_readvariableop3savev2_adam_m_6__to__m_7_bias_v_read_readvariableop5savev2_adam_m_7__to__m_8_kernel_v_read_readvariableop3savev2_adam_m_7__to__m_8_bias_v_read_readvariableop5savev2_adam_m_8__to__m_9_kernel_v_read_readvariableop3savev2_adam_m_8__to__m_9_bias_v_read_readvariableop6savev2_adam_m_9__to__m_10_kernel_v_read_readvariableop4savev2_adam_m_9__to__m_10_bias_v_read_readvariableop7savev2_adam_m_10__to__m_11_kernel_v_read_readvariableop5savev2_adam_m_10__to__m_11_bias_v_read_readvariableop7savev2_adam_m_11__to__m_12_kernel_v_read_readvariableop5savev2_adam_m_11__to__m_12_bias_v_read_readvariableop7savev2_adam_m_12__to__m_13_kernel_v_read_readvariableop5savev2_adam_m_12__to__m_13_bias_v_read_readvariableop7savev2_adam_m_13__to__m_14_kernel_v_read_readvariableop5savev2_adam_m_13__to__m_14_bias_v_read_readvariableop7savev2_adam_m_14__to__m_15_kernel_v_read_readvariableop5savev2_adam_m_14__to__m_15_bias_v_read_readvariableop7savev2_adam_m_15__to__m_16_kernel_v_read_readvariableop5savev2_adam_m_15__to__m_16_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *v
dtypesl
j2h	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>:: : : : : : : :>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>::>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>>:>:>:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$	 

_output_shapes

:>>: 


_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>>: 

_output_shapes
:>:$ 

_output_shapes

:>:  

_output_shapes
::!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :$

_output_shapes
: :%

_output_shapes
: :&

_output_shapes
: :'

_output_shapes
: :$( 

_output_shapes

:>: )

_output_shapes
:>:$* 

_output_shapes

:>>: +

_output_shapes
:>:$, 

_output_shapes

:>>: -

_output_shapes
:>:$. 

_output_shapes

:>>: /

_output_shapes
:>:$0 

_output_shapes

:>>: 1

_output_shapes
:>:$2 

_output_shapes

:>>: 3

_output_shapes
:>:$4 

_output_shapes

:>>: 5

_output_shapes
:>:$6 

_output_shapes

:>>: 7

_output_shapes
:>:$8 

_output_shapes

:>>: 9

_output_shapes
:>:$: 

_output_shapes

:>>: ;

_output_shapes
:>:$< 

_output_shapes

:>>: =

_output_shapes
:>:$> 

_output_shapes

:>>: ?

_output_shapes
:>:$@ 

_output_shapes

:>>: A

_output_shapes
:>:$B 

_output_shapes

:>>: C

_output_shapes
:>:$D 

_output_shapes

:>>: E

_output_shapes
:>:$F 

_output_shapes

:>: G

_output_shapes
::$H 

_output_shapes

:>: I

_output_shapes
:>:$J 

_output_shapes

:>>: K

_output_shapes
:>:$L 

_output_shapes

:>>: M

_output_shapes
:>:$N 

_output_shapes

:>>: O

_output_shapes
:>:$P 

_output_shapes

:>>: Q

_output_shapes
:>:$R 

_output_shapes

:>>: S

_output_shapes
:>:$T 

_output_shapes

:>>: U

_output_shapes
:>:$V 

_output_shapes

:>>: W

_output_shapes
:>:$X 

_output_shapes

:>>: Y

_output_shapes
:>:$Z 

_output_shapes

:>>: [

_output_shapes
:>:$\ 

_output_shapes

:>>: ]

_output_shapes
:>:$^ 

_output_shapes

:>>: _

_output_shapes
:>:$` 

_output_shapes

:>>: a

_output_shapes
:>:$b 

_output_shapes

:>>: c

_output_shapes
:>:$d 

_output_shapes

:>>: e

_output_shapes
:>:$f 

_output_shapes

:>: g

_output_shapes
::h

_output_shapes
: 
?
?
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_13991905

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_6__to__m_7/kernel/Regularizer/Abs?
%m_6__to__m_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_6__to__m_7/kernel/Regularizer/Const?
#m_6__to__m_7/kernel/Regularizer/SumSum'm_6__to__m_7/kernel/Regularizer/Abs:y:0.m_6__to__m_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/Sum?
%m_6__to__m_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_6__to__m_7/kernel/Regularizer/mul/x?
#m_6__to__m_7/kernel/Regularizer/mulMul.m_6__to__m_7/kernel/Regularizer/mul/x:output:0,m_6__to__m_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
/__inference_m_4__to__m_5_layer_call_fn_13991850

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_139896252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
__inference_loss_fn_11_13992334O
=m_11__to__m_12_kernel_regularizer_abs_readvariableop_resource:>>
identity??4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_11__to__m_12_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_11__to__m_12/kernel/Regularizer/Abs?
'm_11__to__m_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_11__to__m_12/kernel/Regularizer/Const?
%m_11__to__m_12/kernel/Regularizer/SumSum)m_11__to__m_12/kernel/Regularizer/Abs:y:00m_11__to__m_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/Sum?
'm_11__to__m_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_11__to__m_12/kernel/Regularizer/mul/x?
%m_11__to__m_12/kernel/Regularizer/mulMul0m_11__to__m_12/kernel/Regularizer/mul/x:output:0.m_11__to__m_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/mul?
IdentityIdentity)m_11__to__m_12/kernel/Regularizer/mul:z:05^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp
?
?
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_13991809

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_3__to__m_4/kernel/Regularizer/Abs?
%m_3__to__m_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_3__to__m_4/kernel/Regularizer/Const?
#m_3__to__m_4/kernel/Regularizer/SumSum'm_3__to__m_4/kernel/Regularizer/Abs:y:0.m_3__to__m_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/Sum?
%m_3__to__m_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_3__to__m_4/kernel/Regularizer/mul/x?
#m_3__to__m_4/kernel/Regularizer/mulMul.m_3__to__m_4/kernel/Regularizer/mul/x:output:0,m_3__to__m_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
??
?
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13990775
m_0__to__m_1_input'
m_0__to__m_1_13990598:>#
m_0__to__m_1_13990600:>'
m_1__to__m_2_13990603:>>#
m_1__to__m_2_13990605:>'
m_2__to__m_3_13990608:>>#
m_2__to__m_3_13990610:>'
m_3__to__m_4_13990613:>>#
m_3__to__m_4_13990615:>'
m_4__to__m_5_13990618:>>#
m_4__to__m_5_13990620:>'
m_5__to__m_6_13990623:>>#
m_5__to__m_6_13990625:>'
m_6__to__m_7_13990628:>>#
m_6__to__m_7_13990630:>'
m_7__to__m_8_13990633:>>#
m_7__to__m_8_13990635:>'
m_8__to__m_9_13990638:>>#
m_8__to__m_9_13990640:>(
m_9__to__m_10_13990643:>>$
m_9__to__m_10_13990645:>)
m_10__to__m_11_13990648:>>%
m_10__to__m_11_13990650:>)
m_11__to__m_12_13990653:>>%
m_11__to__m_12_13990655:>)
m_12__to__m_13_13990658:>>%
m_12__to__m_13_13990660:>)
m_13__to__m_14_13990663:>>%
m_13__to__m_14_13990665:>)
m_14__to__m_15_13990668:>>%
m_14__to__m_15_13990670:>)
m_15__to__m_16_13990673:>%
m_15__to__m_16_13990675:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?&m_15__to__m_16/StatefulPartitionedCall?4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallm_0__to__m_1_inputm_0__to__m_1_13990598m_0__to__m_1_13990600*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_139895332&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_13990603m_1__to__m_2_13990605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_139895562&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_13990608m_2__to__m_3_13990610*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_139895792&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_13990613m_3__to__m_4_13990615*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_139896022&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_13990618m_4__to__m_5_13990620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_139896252&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_13990623m_5__to__m_6_13990625*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_139896482&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_13990628m_6__to__m_7_13990630*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_139896712&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_13990633m_7__to__m_8_13990635*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_139896942&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_13990638m_8__to__m_9_13990640*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_139897172&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_13990643m_9__to__m_10_13990645*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_139897402'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_13990648m_10__to__m_11_13990650*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_139897632(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_13990653m_11__to__m_12_13990655*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_139897862(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_13990658m_12__to__m_13_13990660*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_139898092(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_13990663m_13__to__m_14_13990665*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_139898322(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_13990668m_14__to__m_15_13990670*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_139898552(
&m_14__to__m_15/StatefulPartitionedCall?
&m_15__to__m_16/StatefulPartitionedCallStatefulPartitionedCall/m_14__to__m_15/StatefulPartitionedCall:output:0m_15__to__m_16_13990673m_15__to__m_16_13990675*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_15__to__m_16_layer_call_and_return_conditional_losses_139898782(
&m_15__to__m_16/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_13990598*
_output_shapes

:>*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2%
#m_0__to__m_1/kernel/Regularizer/Abs?
%m_0__to__m_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_0__to__m_1/kernel/Regularizer/Const?
#m_0__to__m_1/kernel/Regularizer/SumSum'm_0__to__m_1/kernel/Regularizer/Abs:y:0.m_0__to__m_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/Sum?
%m_0__to__m_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_0__to__m_1/kernel/Regularizer/mul/x?
#m_0__to__m_1/kernel/Regularizer/mulMul.m_0__to__m_1/kernel/Regularizer/mul/x:output:0,m_0__to__m_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/mul?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_13990603*
_output_shapes

:>>*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_1__to__m_2/kernel/Regularizer/Abs?
%m_1__to__m_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_1__to__m_2/kernel/Regularizer/Const?
#m_1__to__m_2/kernel/Regularizer/SumSum'm_1__to__m_2/kernel/Regularizer/Abs:y:0.m_1__to__m_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/Sum?
%m_1__to__m_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_1__to__m_2/kernel/Regularizer/mul/x?
#m_1__to__m_2/kernel/Regularizer/mulMul.m_1__to__m_2/kernel/Regularizer/mul/x:output:0,m_1__to__m_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/mul?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_13990608*
_output_shapes

:>>*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_2__to__m_3/kernel/Regularizer/Abs?
%m_2__to__m_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_2__to__m_3/kernel/Regularizer/Const?
#m_2__to__m_3/kernel/Regularizer/SumSum'm_2__to__m_3/kernel/Regularizer/Abs:y:0.m_2__to__m_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/Sum?
%m_2__to__m_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_2__to__m_3/kernel/Regularizer/mul/x?
#m_2__to__m_3/kernel/Regularizer/mulMul.m_2__to__m_3/kernel/Regularizer/mul/x:output:0,m_2__to__m_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/mul?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_13990613*
_output_shapes

:>>*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_3__to__m_4/kernel/Regularizer/Abs?
%m_3__to__m_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_3__to__m_4/kernel/Regularizer/Const?
#m_3__to__m_4/kernel/Regularizer/SumSum'm_3__to__m_4/kernel/Regularizer/Abs:y:0.m_3__to__m_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/Sum?
%m_3__to__m_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_3__to__m_4/kernel/Regularizer/mul/x?
#m_3__to__m_4/kernel/Regularizer/mulMul.m_3__to__m_4/kernel/Regularizer/mul/x:output:0,m_3__to__m_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/mul?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_13990618*
_output_shapes

:>>*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_4__to__m_5/kernel/Regularizer/Abs?
%m_4__to__m_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_4__to__m_5/kernel/Regularizer/Const?
#m_4__to__m_5/kernel/Regularizer/SumSum'm_4__to__m_5/kernel/Regularizer/Abs:y:0.m_4__to__m_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/Sum?
%m_4__to__m_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_4__to__m_5/kernel/Regularizer/mul/x?
#m_4__to__m_5/kernel/Regularizer/mulMul.m_4__to__m_5/kernel/Regularizer/mul/x:output:0,m_4__to__m_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/mul?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_13990623*
_output_shapes

:>>*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_5__to__m_6/kernel/Regularizer/Abs?
%m_5__to__m_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_5__to__m_6/kernel/Regularizer/Const?
#m_5__to__m_6/kernel/Regularizer/SumSum'm_5__to__m_6/kernel/Regularizer/Abs:y:0.m_5__to__m_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/Sum?
%m_5__to__m_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_5__to__m_6/kernel/Regularizer/mul/x?
#m_5__to__m_6/kernel/Regularizer/mulMul.m_5__to__m_6/kernel/Regularizer/mul/x:output:0,m_5__to__m_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/mul?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_13990628*
_output_shapes

:>>*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_6__to__m_7/kernel/Regularizer/Abs?
%m_6__to__m_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_6__to__m_7/kernel/Regularizer/Const?
#m_6__to__m_7/kernel/Regularizer/SumSum'm_6__to__m_7/kernel/Regularizer/Abs:y:0.m_6__to__m_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/Sum?
%m_6__to__m_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_6__to__m_7/kernel/Regularizer/mul/x?
#m_6__to__m_7/kernel/Regularizer/mulMul.m_6__to__m_7/kernel/Regularizer/mul/x:output:0,m_6__to__m_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/mul?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_13990633*
_output_shapes

:>>*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_7__to__m_8/kernel/Regularizer/Abs?
%m_7__to__m_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_7__to__m_8/kernel/Regularizer/Const?
#m_7__to__m_8/kernel/Regularizer/SumSum'm_7__to__m_8/kernel/Regularizer/Abs:y:0.m_7__to__m_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/Sum?
%m_7__to__m_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_7__to__m_8/kernel/Regularizer/mul/x?
#m_7__to__m_8/kernel/Regularizer/mulMul.m_7__to__m_8/kernel/Regularizer/mul/x:output:0,m_7__to__m_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/mul?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_13990638*
_output_shapes

:>>*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_8__to__m_9/kernel/Regularizer/Abs?
%m_8__to__m_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_8__to__m_9/kernel/Regularizer/Const?
#m_8__to__m_9/kernel/Regularizer/SumSum'm_8__to__m_9/kernel/Regularizer/Abs:y:0.m_8__to__m_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/Sum?
%m_8__to__m_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_8__to__m_9/kernel/Regularizer/mul/x?
#m_8__to__m_9/kernel/Regularizer/mulMul.m_8__to__m_9/kernel/Regularizer/mul/x:output:0,m_8__to__m_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/mul?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_13990643*
_output_shapes

:>>*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2&
$m_9__to__m_10/kernel/Regularizer/Abs?
&m_9__to__m_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&m_9__to__m_10/kernel/Regularizer/Const?
$m_9__to__m_10/kernel/Regularizer/SumSum(m_9__to__m_10/kernel/Regularizer/Abs:y:0/m_9__to__m_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/Sum?
&m_9__to__m_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92(
&m_9__to__m_10/kernel/Regularizer/mul/x?
$m_9__to__m_10/kernel/Regularizer/mulMul/m_9__to__m_10/kernel/Regularizer/mul/x:output:0-m_9__to__m_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/mul?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_13990648*
_output_shapes

:>>*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_10__to__m_11/kernel/Regularizer/Abs?
'm_10__to__m_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_10__to__m_11/kernel/Regularizer/Const?
%m_10__to__m_11/kernel/Regularizer/SumSum)m_10__to__m_11/kernel/Regularizer/Abs:y:00m_10__to__m_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/Sum?
'm_10__to__m_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_10__to__m_11/kernel/Regularizer/mul/x?
%m_10__to__m_11/kernel/Regularizer/mulMul0m_10__to__m_11/kernel/Regularizer/mul/x:output:0.m_10__to__m_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/mul?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_13990653*
_output_shapes

:>>*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_11__to__m_12/kernel/Regularizer/Abs?
'm_11__to__m_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_11__to__m_12/kernel/Regularizer/Const?
%m_11__to__m_12/kernel/Regularizer/SumSum)m_11__to__m_12/kernel/Regularizer/Abs:y:00m_11__to__m_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/Sum?
'm_11__to__m_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_11__to__m_12/kernel/Regularizer/mul/x?
%m_11__to__m_12/kernel/Regularizer/mulMul0m_11__to__m_12/kernel/Regularizer/mul/x:output:0.m_11__to__m_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/mul?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_13990658*
_output_shapes

:>>*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_12__to__m_13/kernel/Regularizer/Abs?
'm_12__to__m_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_12__to__m_13/kernel/Regularizer/Const?
%m_12__to__m_13/kernel/Regularizer/SumSum)m_12__to__m_13/kernel/Regularizer/Abs:y:00m_12__to__m_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/Sum?
'm_12__to__m_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_12__to__m_13/kernel/Regularizer/mul/x?
%m_12__to__m_13/kernel/Regularizer/mulMul0m_12__to__m_13/kernel/Regularizer/mul/x:output:0.m_12__to__m_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/mul?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_13990663*
_output_shapes

:>>*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_13__to__m_14/kernel/Regularizer/Abs?
'm_13__to__m_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_13__to__m_14/kernel/Regularizer/Const?
%m_13__to__m_14/kernel/Regularizer/SumSum)m_13__to__m_14/kernel/Regularizer/Abs:y:00m_13__to__m_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/Sum?
'm_13__to__m_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_13__to__m_14/kernel/Regularizer/mul/x?
%m_13__to__m_14/kernel/Regularizer/mulMul0m_13__to__m_14/kernel/Regularizer/mul/x:output:0.m_13__to__m_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/mul?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_13990668*
_output_shapes

:>>*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_14__to__m_15/kernel/Regularizer/Abs?
'm_14__to__m_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_14__to__m_15/kernel/Regularizer/Const?
%m_14__to__m_15/kernel/Regularizer/SumSum)m_14__to__m_15/kernel/Regularizer/Abs:y:00m_14__to__m_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/Sum?
'm_14__to__m_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_14__to__m_15/kernel/Regularizer/mul/x?
%m_14__to__m_15/kernel/Regularizer/mulMul0m_14__to__m_15/kernel/Regularizer/mul/x:output:0.m_14__to__m_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/mul?
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_15__to__m_16_13990673*
_output_shapes

:>*
dtype026
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
%m_15__to__m_16/kernel/Regularizer/AbsAbs<m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2'
%m_15__to__m_16/kernel/Regularizer/Abs?
'm_15__to__m_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_15__to__m_16/kernel/Regularizer/Const?
%m_15__to__m_16/kernel/Regularizer/SumSum)m_15__to__m_16/kernel/Regularizer/Abs:y:00m_15__to__m_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/Sum?
'm_15__to__m_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_15__to__m_16/kernel/Regularizer/mul/x?
%m_15__to__m_16/kernel/Regularizer/mulMul0m_15__to__m_16/kernel/Regularizer/mul/x:output:0.m_15__to__m_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/mul?
IdentityIdentity/m_15__to__m_16/StatefulPartitionedCall:output:0%^m_0__to__m_1/StatefulPartitionedCall3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp'^m_10__to__m_11/StatefulPartitionedCall5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp'^m_11__to__m_12/StatefulPartitionedCall5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp'^m_12__to__m_13/StatefulPartitionedCall5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp'^m_13__to__m_14/StatefulPartitionedCall5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp'^m_14__to__m_15/StatefulPartitionedCall5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp'^m_15__to__m_16/StatefulPartitionedCall5^m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp%^m_1__to__m_2/StatefulPartitionedCall3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp%^m_2__to__m_3/StatefulPartitionedCall3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp%^m_3__to__m_4/StatefulPartitionedCall3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp%^m_4__to__m_5/StatefulPartitionedCall3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp%^m_5__to__m_6/StatefulPartitionedCall3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp%^m_6__to__m_7/StatefulPartitionedCall3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp%^m_7__to__m_8/StatefulPartitionedCall3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp%^m_8__to__m_9/StatefulPartitionedCall3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp&^m_9__to__m_10/StatefulPartitionedCall4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$m_0__to__m_1/StatefulPartitionedCall$m_0__to__m_1/StatefulPartitionedCall2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2P
&m_10__to__m_11/StatefulPartitionedCall&m_10__to__m_11/StatefulPartitionedCall2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp2P
&m_11__to__m_12/StatefulPartitionedCall&m_11__to__m_12/StatefulPartitionedCall2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp2P
&m_12__to__m_13/StatefulPartitionedCall&m_12__to__m_13/StatefulPartitionedCall2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp2P
&m_13__to__m_14/StatefulPartitionedCall&m_13__to__m_14/StatefulPartitionedCall2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp2P
&m_14__to__m_15/StatefulPartitionedCall&m_14__to__m_15/StatefulPartitionedCall2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2P
&m_15__to__m_16/StatefulPartitionedCall&m_15__to__m_16/StatefulPartitionedCall2l
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp2L
$m_1__to__m_2/StatefulPartitionedCall$m_1__to__m_2/StatefulPartitionedCall2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2L
$m_2__to__m_3/StatefulPartitionedCall$m_2__to__m_3/StatefulPartitionedCall2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2L
$m_3__to__m_4/StatefulPartitionedCall$m_3__to__m_4/StatefulPartitionedCall2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2L
$m_4__to__m_5/StatefulPartitionedCall$m_4__to__m_5/StatefulPartitionedCall2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2L
$m_5__to__m_6/StatefulPartitionedCall$m_5__to__m_6/StatefulPartitionedCall2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2L
$m_6__to__m_7/StatefulPartitionedCall$m_6__to__m_7/StatefulPartitionedCall2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2L
$m_7__to__m_8/StatefulPartitionedCall$m_7__to__m_8/StatefulPartitionedCall2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2L
$m_8__to__m_9/StatefulPartitionedCall$m_8__to__m_9/StatefulPartitionedCall2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2N
%m_9__to__m_10/StatefulPartitionedCall%m_9__to__m_10/StatefulPartitionedCall2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_13992065

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_11__to__m_12/kernel/Regularizer/Abs?
'm_11__to__m_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_11__to__m_12/kernel/Regularizer/Const?
%m_11__to__m_12/kernel/Regularizer/SumSum)m_11__to__m_12/kernel/Regularizer/Abs:y:00m_11__to__m_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/Sum?
'm_11__to__m_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_11__to__m_12/kernel/Regularizer/mul/x?
%m_11__to__m_12/kernel/Regularizer/mulMul0m_11__to__m_12/kernel/Regularizer/mul/x:output:0.m_11__to__m_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_13989786

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_11__to__m_12/kernel/Regularizer/Abs?
'm_11__to__m_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_11__to__m_12/kernel/Regularizer/Const?
%m_11__to__m_12/kernel/Regularizer/SumSum)m_11__to__m_12/kernel/Regularizer/Abs:y:00m_11__to__m_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/Sum?
'm_11__to__m_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_11__to__m_12/kernel/Regularizer/mul/x?
%m_11__to__m_12/kernel/Regularizer/mulMul0m_11__to__m_12/kernel/Regularizer/mul/x:output:0.m_11__to__m_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_13989832

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_13__to__m_14/kernel/Regularizer/Abs?
'm_13__to__m_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_13__to__m_14/kernel/Regularizer/Const?
%m_13__to__m_14/kernel/Regularizer/SumSum)m_13__to__m_14/kernel/Regularizer/Abs:y:00m_13__to__m_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/Sum?
'm_13__to__m_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_13__to__m_14/kernel/Regularizer/mul/x?
%m_13__to__m_14/kernel/Regularizer/mulMul0m_13__to__m_14/kernel/Regularizer/mul/x:output:0.m_13__to__m_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_13991937

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_7__to__m_8/kernel/Regularizer/Abs?
%m_7__to__m_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_7__to__m_8/kernel/Regularizer/Const?
#m_7__to__m_8/kernel/Regularizer/SumSum'm_7__to__m_8/kernel/Regularizer/Abs:y:0.m_7__to__m_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/Sum?
%m_7__to__m_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_7__to__m_8/kernel/Regularizer/mul/x?
#m_7__to__m_8/kernel/Regularizer/mulMul.m_7__to__m_8/kernel/Regularizer/mul/x:output:0,m_7__to__m_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
/__inference_m_0__to__m_1_layer_call_fn_13991722

inputs
unknown:>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_139895332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_13989717

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_8__to__m_9/kernel/Regularizer/Abs?
%m_8__to__m_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_8__to__m_9/kernel/Regularizer/Const?
#m_8__to__m_9/kernel/Regularizer/SumSum'm_8__to__m_9/kernel/Regularizer/Abs:y:0.m_8__to__m_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/Sum?
%m_8__to__m_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_8__to__m_9/kernel/Regularizer/mul/x?
#m_8__to__m_9/kernel/Regularizer/mulMul.m_8__to__m_9/kernel/Regularizer/mul/x:output:0,m_8__to__m_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_13991841

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_4__to__m_5/kernel/Regularizer/Abs?
%m_4__to__m_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_4__to__m_5/kernel/Regularizer/Const?
#m_4__to__m_5/kernel/Regularizer/SumSum'm_4__to__m_5/kernel/Regularizer/Abs:y:0.m_4__to__m_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/Sum?
%m_4__to__m_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_4__to__m_5/kernel/Regularizer/mul/x?
#m_4__to__m_5/kernel/Regularizer/mulMul.m_4__to__m_5/kernel/Regularizer/mul/x:output:0,m_4__to__m_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_13991873

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_5__to__m_6/kernel/Regularizer/Abs?
%m_5__to__m_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_5__to__m_6/kernel/Regularizer/Const?
#m_5__to__m_6/kernel/Regularizer/SumSum'm_5__to__m_6/kernel/Regularizer/Abs:y:0.m_5__to__m_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/Sum?
%m_5__to__m_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_5__to__m_6/kernel/Regularizer/mul/x?
#m_5__to__m_6/kernel/Regularizer/mulMul.m_5__to__m_6/kernel/Regularizer/mul/x:output:0,m_5__to__m_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_13992097

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_12__to__m_13/kernel/Regularizer/Abs?
'm_12__to__m_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_12__to__m_13/kernel/Regularizer/Const?
%m_12__to__m_13/kernel/Regularizer/SumSum)m_12__to__m_13/kernel/Regularizer/Abs:y:00m_12__to__m_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/Sum?
'm_12__to__m_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_12__to__m_13/kernel/Regularizer/mul/x?
%m_12__to__m_13/kernel/Regularizer/mulMul0m_12__to__m_13/kernel/Regularizer/mul/x:output:0.m_12__to__m_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
1__inference_m_14__to__m_15_layer_call_fn_13992170

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_139898552
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
__inference_loss_fn_6_13992279M
;m_6__to__m_7_kernel_regularizer_abs_readvariableop_resource:>>
identity??2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_6__to__m_7_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_6__to__m_7/kernel/Regularizer/Abs?
%m_6__to__m_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_6__to__m_7/kernel/Regularizer/Const?
#m_6__to__m_7/kernel/Regularizer/SumSum'm_6__to__m_7/kernel/Regularizer/Abs:y:0.m_6__to__m_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/Sum?
%m_6__to__m_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_6__to__m_7/kernel/Regularizer/mul/x?
#m_6__to__m_7/kernel/Regularizer/mulMul.m_6__to__m_7/kernel/Regularizer/mul/x:output:0,m_6__to__m_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/mul?
IdentityIdentity'm_6__to__m_7/kernel/Regularizer/mul:z:03^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp
?
?
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_13992129

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_13__to__m_14/kernel/Regularizer/Abs?
'm_13__to__m_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_13__to__m_14/kernel/Regularizer/Const?
%m_13__to__m_14/kernel/Regularizer/SumSum)m_13__to__m_14/kernel/Regularizer/Abs:y:00m_13__to__m_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/Sum?
'm_13__to__m_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_13__to__m_14/kernel/Regularizer/mul/x?
%m_13__to__m_14/kernel/Regularizer/mulMul0m_13__to__m_14/kernel/Regularizer/mul/x:output:0.m_13__to__m_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
__inference_loss_fn_15_13992378O
=m_15__to__m_16_kernel_regularizer_abs_readvariableop_resource:>
identity??4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_15__to__m_16_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>*
dtype026
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
%m_15__to__m_16/kernel/Regularizer/AbsAbs<m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2'
%m_15__to__m_16/kernel/Regularizer/Abs?
'm_15__to__m_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_15__to__m_16/kernel/Regularizer/Const?
%m_15__to__m_16/kernel/Regularizer/SumSum)m_15__to__m_16/kernel/Regularizer/Abs:y:00m_15__to__m_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/Sum?
'm_15__to__m_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_15__to__m_16/kernel/Regularizer/mul/x?
%m_15__to__m_16/kernel/Regularizer/mulMul0m_15__to__m_16/kernel/Regularizer/mul/x:output:0.m_15__to__m_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/mul?
IdentityIdentity)m_15__to__m_16/kernel/Regularizer/mul:z:05^m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp
??
?B
$__inference__traced_restore_13993029
file_prefix6
$assignvariableop_m_0__to__m_1_kernel:>2
$assignvariableop_1_m_0__to__m_1_bias:>8
&assignvariableop_2_m_1__to__m_2_kernel:>>2
$assignvariableop_3_m_1__to__m_2_bias:>8
&assignvariableop_4_m_2__to__m_3_kernel:>>2
$assignvariableop_5_m_2__to__m_3_bias:>8
&assignvariableop_6_m_3__to__m_4_kernel:>>2
$assignvariableop_7_m_3__to__m_4_bias:>8
&assignvariableop_8_m_4__to__m_5_kernel:>>2
$assignvariableop_9_m_4__to__m_5_bias:>9
'assignvariableop_10_m_5__to__m_6_kernel:>>3
%assignvariableop_11_m_5__to__m_6_bias:>9
'assignvariableop_12_m_6__to__m_7_kernel:>>3
%assignvariableop_13_m_6__to__m_7_bias:>9
'assignvariableop_14_m_7__to__m_8_kernel:>>3
%assignvariableop_15_m_7__to__m_8_bias:>9
'assignvariableop_16_m_8__to__m_9_kernel:>>3
%assignvariableop_17_m_8__to__m_9_bias:>:
(assignvariableop_18_m_9__to__m_10_kernel:>>4
&assignvariableop_19_m_9__to__m_10_bias:>;
)assignvariableop_20_m_10__to__m_11_kernel:>>5
'assignvariableop_21_m_10__to__m_11_bias:>;
)assignvariableop_22_m_11__to__m_12_kernel:>>5
'assignvariableop_23_m_11__to__m_12_bias:>;
)assignvariableop_24_m_12__to__m_13_kernel:>>5
'assignvariableop_25_m_12__to__m_13_bias:>;
)assignvariableop_26_m_13__to__m_14_kernel:>>5
'assignvariableop_27_m_13__to__m_14_bias:>;
)assignvariableop_28_m_14__to__m_15_kernel:>>5
'assignvariableop_29_m_14__to__m_15_bias:>;
)assignvariableop_30_m_15__to__m_16_kernel:>5
'assignvariableop_31_m_15__to__m_16_bias:'
assignvariableop_32_adam_iter:	 )
assignvariableop_33_adam_beta_1: )
assignvariableop_34_adam_beta_2: (
assignvariableop_35_adam_decay: 0
&assignvariableop_36_adam_learning_rate: #
assignvariableop_37_total: #
assignvariableop_38_count: @
.assignvariableop_39_adam_m_0__to__m_1_kernel_m:>:
,assignvariableop_40_adam_m_0__to__m_1_bias_m:>@
.assignvariableop_41_adam_m_1__to__m_2_kernel_m:>>:
,assignvariableop_42_adam_m_1__to__m_2_bias_m:>@
.assignvariableop_43_adam_m_2__to__m_3_kernel_m:>>:
,assignvariableop_44_adam_m_2__to__m_3_bias_m:>@
.assignvariableop_45_adam_m_3__to__m_4_kernel_m:>>:
,assignvariableop_46_adam_m_3__to__m_4_bias_m:>@
.assignvariableop_47_adam_m_4__to__m_5_kernel_m:>>:
,assignvariableop_48_adam_m_4__to__m_5_bias_m:>@
.assignvariableop_49_adam_m_5__to__m_6_kernel_m:>>:
,assignvariableop_50_adam_m_5__to__m_6_bias_m:>@
.assignvariableop_51_adam_m_6__to__m_7_kernel_m:>>:
,assignvariableop_52_adam_m_6__to__m_7_bias_m:>@
.assignvariableop_53_adam_m_7__to__m_8_kernel_m:>>:
,assignvariableop_54_adam_m_7__to__m_8_bias_m:>@
.assignvariableop_55_adam_m_8__to__m_9_kernel_m:>>:
,assignvariableop_56_adam_m_8__to__m_9_bias_m:>A
/assignvariableop_57_adam_m_9__to__m_10_kernel_m:>>;
-assignvariableop_58_adam_m_9__to__m_10_bias_m:>B
0assignvariableop_59_adam_m_10__to__m_11_kernel_m:>><
.assignvariableop_60_adam_m_10__to__m_11_bias_m:>B
0assignvariableop_61_adam_m_11__to__m_12_kernel_m:>><
.assignvariableop_62_adam_m_11__to__m_12_bias_m:>B
0assignvariableop_63_adam_m_12__to__m_13_kernel_m:>><
.assignvariableop_64_adam_m_12__to__m_13_bias_m:>B
0assignvariableop_65_adam_m_13__to__m_14_kernel_m:>><
.assignvariableop_66_adam_m_13__to__m_14_bias_m:>B
0assignvariableop_67_adam_m_14__to__m_15_kernel_m:>><
.assignvariableop_68_adam_m_14__to__m_15_bias_m:>B
0assignvariableop_69_adam_m_15__to__m_16_kernel_m:><
.assignvariableop_70_adam_m_15__to__m_16_bias_m:@
.assignvariableop_71_adam_m_0__to__m_1_kernel_v:>:
,assignvariableop_72_adam_m_0__to__m_1_bias_v:>@
.assignvariableop_73_adam_m_1__to__m_2_kernel_v:>>:
,assignvariableop_74_adam_m_1__to__m_2_bias_v:>@
.assignvariableop_75_adam_m_2__to__m_3_kernel_v:>>:
,assignvariableop_76_adam_m_2__to__m_3_bias_v:>@
.assignvariableop_77_adam_m_3__to__m_4_kernel_v:>>:
,assignvariableop_78_adam_m_3__to__m_4_bias_v:>@
.assignvariableop_79_adam_m_4__to__m_5_kernel_v:>>:
,assignvariableop_80_adam_m_4__to__m_5_bias_v:>@
.assignvariableop_81_adam_m_5__to__m_6_kernel_v:>>:
,assignvariableop_82_adam_m_5__to__m_6_bias_v:>@
.assignvariableop_83_adam_m_6__to__m_7_kernel_v:>>:
,assignvariableop_84_adam_m_6__to__m_7_bias_v:>@
.assignvariableop_85_adam_m_7__to__m_8_kernel_v:>>:
,assignvariableop_86_adam_m_7__to__m_8_bias_v:>@
.assignvariableop_87_adam_m_8__to__m_9_kernel_v:>>:
,assignvariableop_88_adam_m_8__to__m_9_bias_v:>A
/assignvariableop_89_adam_m_9__to__m_10_kernel_v:>>;
-assignvariableop_90_adam_m_9__to__m_10_bias_v:>B
0assignvariableop_91_adam_m_10__to__m_11_kernel_v:>><
.assignvariableop_92_adam_m_10__to__m_11_bias_v:>B
0assignvariableop_93_adam_m_11__to__m_12_kernel_v:>><
.assignvariableop_94_adam_m_11__to__m_12_bias_v:>B
0assignvariableop_95_adam_m_12__to__m_13_kernel_v:>><
.assignvariableop_96_adam_m_12__to__m_13_bias_v:>B
0assignvariableop_97_adam_m_13__to__m_14_kernel_v:>><
.assignvariableop_98_adam_m_13__to__m_14_bias_v:>B
0assignvariableop_99_adam_m_14__to__m_15_kernel_v:>>=
/assignvariableop_100_adam_m_14__to__m_15_bias_v:>C
1assignvariableop_101_adam_m_15__to__m_16_kernel_v:>=
/assignvariableop_102_adam_m_15__to__m_16_bias_v:
identity_104??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_100?AssignVariableOp_101?AssignVariableOp_102?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?AssignVariableOp_97?AssignVariableOp_98?AssignVariableOp_99?;
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*?:
value?:B?:hB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-15/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-15/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:h*
dtype0*?
value?B?hB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*v
dtypesl
j2h	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp$assignvariableop_m_0__to__m_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp$assignvariableop_1_m_0__to__m_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp&assignvariableop_2_m_1__to__m_2_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp$assignvariableop_3_m_1__to__m_2_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp&assignvariableop_4_m_2__to__m_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp$assignvariableop_5_m_2__to__m_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp&assignvariableop_6_m_3__to__m_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp$assignvariableop_7_m_3__to__m_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp&assignvariableop_8_m_4__to__m_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp$assignvariableop_9_m_4__to__m_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp'assignvariableop_10_m_5__to__m_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp%assignvariableop_11_m_5__to__m_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp'assignvariableop_12_m_6__to__m_7_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp%assignvariableop_13_m_6__to__m_7_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp'assignvariableop_14_m_7__to__m_8_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp%assignvariableop_15_m_7__to__m_8_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp'assignvariableop_16_m_8__to__m_9_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp%assignvariableop_17_m_8__to__m_9_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp(assignvariableop_18_m_9__to__m_10_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp&assignvariableop_19_m_9__to__m_10_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp)assignvariableop_20_m_10__to__m_11_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp'assignvariableop_21_m_10__to__m_11_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp)assignvariableop_22_m_11__to__m_12_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp'assignvariableop_23_m_11__to__m_12_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp)assignvariableop_24_m_12__to__m_13_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp'assignvariableop_25_m_12__to__m_13_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp)assignvariableop_26_m_13__to__m_14_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp'assignvariableop_27_m_13__to__m_14_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp)assignvariableop_28_m_14__to__m_15_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp'assignvariableop_29_m_14__to__m_15_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp)assignvariableop_30_m_15__to__m_16_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp'assignvariableop_31_m_15__to__m_16_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_iterIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_beta_1Identity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpassignvariableop_34_adam_beta_2Identity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_adam_decayIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_learning_rateIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpassignvariableop_37_totalIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpassignvariableop_38_countIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp.assignvariableop_39_adam_m_0__to__m_1_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp,assignvariableop_40_adam_m_0__to__m_1_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp.assignvariableop_41_adam_m_1__to__m_2_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_m_1__to__m_2_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp.assignvariableop_43_adam_m_2__to__m_3_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_m_2__to__m_3_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp.assignvariableop_45_adam_m_3__to__m_4_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_m_3__to__m_4_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp.assignvariableop_47_adam_m_4__to__m_5_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp,assignvariableop_48_adam_m_4__to__m_5_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp.assignvariableop_49_adam_m_5__to__m_6_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp,assignvariableop_50_adam_m_5__to__m_6_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp.assignvariableop_51_adam_m_6__to__m_7_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp,assignvariableop_52_adam_m_6__to__m_7_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp.assignvariableop_53_adam_m_7__to__m_8_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp,assignvariableop_54_adam_m_7__to__m_8_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp.assignvariableop_55_adam_m_8__to__m_9_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp,assignvariableop_56_adam_m_8__to__m_9_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp/assignvariableop_57_adam_m_9__to__m_10_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp-assignvariableop_58_adam_m_9__to__m_10_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp0assignvariableop_59_adam_m_10__to__m_11_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp.assignvariableop_60_adam_m_10__to__m_11_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp0assignvariableop_61_adam_m_11__to__m_12_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp.assignvariableop_62_adam_m_11__to__m_12_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp0assignvariableop_63_adam_m_12__to__m_13_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp.assignvariableop_64_adam_m_12__to__m_13_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp0assignvariableop_65_adam_m_13__to__m_14_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp.assignvariableop_66_adam_m_13__to__m_14_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp0assignvariableop_67_adam_m_14__to__m_15_kernel_mIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp.assignvariableop_68_adam_m_14__to__m_15_bias_mIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp0assignvariableop_69_adam_m_15__to__m_16_kernel_mIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp.assignvariableop_70_adam_m_15__to__m_16_bias_mIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp.assignvariableop_71_adam_m_0__to__m_1_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp,assignvariableop_72_adam_m_0__to__m_1_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp.assignvariableop_73_adam_m_1__to__m_2_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp,assignvariableop_74_adam_m_1__to__m_2_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp.assignvariableop_75_adam_m_2__to__m_3_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp,assignvariableop_76_adam_m_2__to__m_3_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp.assignvariableop_77_adam_m_3__to__m_4_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp,assignvariableop_78_adam_m_3__to__m_4_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp.assignvariableop_79_adam_m_4__to__m_5_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp,assignvariableop_80_adam_m_4__to__m_5_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp.assignvariableop_81_adam_m_5__to__m_6_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp,assignvariableop_82_adam_m_5__to__m_6_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp.assignvariableop_83_adam_m_6__to__m_7_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp,assignvariableop_84_adam_m_6__to__m_7_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp.assignvariableop_85_adam_m_7__to__m_8_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp,assignvariableop_86_adam_m_7__to__m_8_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp.assignvariableop_87_adam_m_8__to__m_9_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp,assignvariableop_88_adam_m_8__to__m_9_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp/assignvariableop_89_adam_m_9__to__m_10_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp-assignvariableop_90_adam_m_9__to__m_10_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp0assignvariableop_91_adam_m_10__to__m_11_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp.assignvariableop_92_adam_m_10__to__m_11_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp0assignvariableop_93_adam_m_11__to__m_12_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp.assignvariableop_94_adam_m_11__to__m_12_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp0assignvariableop_95_adam_m_12__to__m_13_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp.assignvariableop_96_adam_m_12__to__m_13_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_96n
Identity_97IdentityRestoreV2:tensors:97"/device:CPU:0*
T0*
_output_shapes
:2
Identity_97?
AssignVariableOp_97AssignVariableOp0assignvariableop_97_adam_m_13__to__m_14_kernel_vIdentity_97:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_97n
Identity_98IdentityRestoreV2:tensors:98"/device:CPU:0*
T0*
_output_shapes
:2
Identity_98?
AssignVariableOp_98AssignVariableOp.assignvariableop_98_adam_m_13__to__m_14_bias_vIdentity_98:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_98n
Identity_99IdentityRestoreV2:tensors:99"/device:CPU:0*
T0*
_output_shapes
:2
Identity_99?
AssignVariableOp_99AssignVariableOp0assignvariableop_99_adam_m_14__to__m_15_kernel_vIdentity_99:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_99q
Identity_100IdentityRestoreV2:tensors:100"/device:CPU:0*
T0*
_output_shapes
:2
Identity_100?
AssignVariableOp_100AssignVariableOp/assignvariableop_100_adam_m_14__to__m_15_bias_vIdentity_100:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_100q
Identity_101IdentityRestoreV2:tensors:101"/device:CPU:0*
T0*
_output_shapes
:2
Identity_101?
AssignVariableOp_101AssignVariableOp1assignvariableop_101_adam_m_15__to__m_16_kernel_vIdentity_101:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_101q
Identity_102IdentityRestoreV2:tensors:102"/device:CPU:0*
T0*
_output_shapes
:2
Identity_102?
AssignVariableOp_102AssignVariableOp/assignvariableop_102_adam_m_15__to__m_16_bias_vIdentity_102:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1029
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_103Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_103?
Identity_104IdentityIdentity_103:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_100^AssignVariableOp_101^AssignVariableOp_102^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^AssignVariableOp_97^AssignVariableOp_98^AssignVariableOp_99*
T0*
_output_shapes
: 2
Identity_104"%
identity_104Identity_104:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102,
AssignVariableOp_100AssignVariableOp_1002,
AssignVariableOp_101AssignVariableOp_1012,
AssignVariableOp_102AssignVariableOp_1022*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862*
AssignVariableOp_87AssignVariableOp_872*
AssignVariableOp_88AssignVariableOp_882*
AssignVariableOp_89AssignVariableOp_892(
AssignVariableOp_9AssignVariableOp_92*
AssignVariableOp_90AssignVariableOp_902*
AssignVariableOp_91AssignVariableOp_912*
AssignVariableOp_92AssignVariableOp_922*
AssignVariableOp_93AssignVariableOp_932*
AssignVariableOp_94AssignVariableOp_942*
AssignVariableOp_95AssignVariableOp_952*
AssignVariableOp_96AssignVariableOp_962*
AssignVariableOp_97AssignVariableOp_972*
AssignVariableOp_98AssignVariableOp_982*
AssignVariableOp_99AssignVariableOp_99:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
? 
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13991340

inputs=
+m_0__to__m_1_matmul_readvariableop_resource:>:
,m_0__to__m_1_biasadd_readvariableop_resource:>=
+m_1__to__m_2_matmul_readvariableop_resource:>>:
,m_1__to__m_2_biasadd_readvariableop_resource:>=
+m_2__to__m_3_matmul_readvariableop_resource:>>:
,m_2__to__m_3_biasadd_readvariableop_resource:>=
+m_3__to__m_4_matmul_readvariableop_resource:>>:
,m_3__to__m_4_biasadd_readvariableop_resource:>=
+m_4__to__m_5_matmul_readvariableop_resource:>>:
,m_4__to__m_5_biasadd_readvariableop_resource:>=
+m_5__to__m_6_matmul_readvariableop_resource:>>:
,m_5__to__m_6_biasadd_readvariableop_resource:>=
+m_6__to__m_7_matmul_readvariableop_resource:>>:
,m_6__to__m_7_biasadd_readvariableop_resource:>=
+m_7__to__m_8_matmul_readvariableop_resource:>>:
,m_7__to__m_8_biasadd_readvariableop_resource:>=
+m_8__to__m_9_matmul_readvariableop_resource:>>:
,m_8__to__m_9_biasadd_readvariableop_resource:>>
,m_9__to__m_10_matmul_readvariableop_resource:>>;
-m_9__to__m_10_biasadd_readvariableop_resource:>?
-m_10__to__m_11_matmul_readvariableop_resource:>><
.m_10__to__m_11_biasadd_readvariableop_resource:>?
-m_11__to__m_12_matmul_readvariableop_resource:>><
.m_11__to__m_12_biasadd_readvariableop_resource:>?
-m_12__to__m_13_matmul_readvariableop_resource:>><
.m_12__to__m_13_biasadd_readvariableop_resource:>?
-m_13__to__m_14_matmul_readvariableop_resource:>><
.m_13__to__m_14_biasadd_readvariableop_resource:>?
-m_14__to__m_15_matmul_readvariableop_resource:>><
.m_14__to__m_15_biasadd_readvariableop_resource:>?
-m_15__to__m_16_matmul_readvariableop_resource:><
.m_15__to__m_16_biasadd_readvariableop_resource:
identity??#m_0__to__m_1/BiasAdd/ReadVariableOp?"m_0__to__m_1/MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?%m_10__to__m_11/BiasAdd/ReadVariableOp?$m_10__to__m_11/MatMul/ReadVariableOp?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?%m_11__to__m_12/BiasAdd/ReadVariableOp?$m_11__to__m_12/MatMul/ReadVariableOp?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?%m_12__to__m_13/BiasAdd/ReadVariableOp?$m_12__to__m_13/MatMul/ReadVariableOp?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?%m_13__to__m_14/BiasAdd/ReadVariableOp?$m_13__to__m_14/MatMul/ReadVariableOp?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?%m_14__to__m_15/BiasAdd/ReadVariableOp?$m_14__to__m_15/MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?%m_15__to__m_16/BiasAdd/ReadVariableOp?$m_15__to__m_16/MatMul/ReadVariableOp?4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?#m_1__to__m_2/BiasAdd/ReadVariableOp?"m_1__to__m_2/MatMul/ReadVariableOp?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?#m_2__to__m_3/BiasAdd/ReadVariableOp?"m_2__to__m_3/MatMul/ReadVariableOp?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?#m_3__to__m_4/BiasAdd/ReadVariableOp?"m_3__to__m_4/MatMul/ReadVariableOp?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?#m_4__to__m_5/BiasAdd/ReadVariableOp?"m_4__to__m_5/MatMul/ReadVariableOp?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?#m_5__to__m_6/BiasAdd/ReadVariableOp?"m_5__to__m_6/MatMul/ReadVariableOp?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?#m_6__to__m_7/BiasAdd/ReadVariableOp?"m_6__to__m_7/MatMul/ReadVariableOp?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?#m_7__to__m_8/BiasAdd/ReadVariableOp?"m_7__to__m_8/MatMul/ReadVariableOp?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?#m_8__to__m_9/BiasAdd/ReadVariableOp?"m_8__to__m_9/MatMul/ReadVariableOp?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?$m_9__to__m_10/BiasAdd/ReadVariableOp?#m_9__to__m_10/MatMul/ReadVariableOp?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
"m_0__to__m_1/MatMul/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:>*
dtype02$
"m_0__to__m_1/MatMul/ReadVariableOp?
m_0__to__m_1/MatMulMatMulinputs*m_0__to__m_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_0__to__m_1/MatMul?
#m_0__to__m_1/BiasAdd/ReadVariableOpReadVariableOp,m_0__to__m_1_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_0__to__m_1/BiasAdd/ReadVariableOp?
m_0__to__m_1/BiasAddAddm_0__to__m_1/MatMul:product:0+m_0__to__m_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_0__to__m_1/BiasAddz
m_0__to__m_1/ReluRelum_0__to__m_1/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_0__to__m_1/Relu?
"m_1__to__m_2/MatMul/ReadVariableOpReadVariableOp+m_1__to__m_2_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_1__to__m_2/MatMul/ReadVariableOp?
m_1__to__m_2/MatMulMatMulm_0__to__m_1/Relu:activations:0*m_1__to__m_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_1__to__m_2/MatMul?
#m_1__to__m_2/BiasAdd/ReadVariableOpReadVariableOp,m_1__to__m_2_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_1__to__m_2/BiasAdd/ReadVariableOp?
m_1__to__m_2/BiasAddAddm_1__to__m_2/MatMul:product:0+m_1__to__m_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_1__to__m_2/BiasAddz
m_1__to__m_2/ReluRelum_1__to__m_2/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_1__to__m_2/Relu?
"m_2__to__m_3/MatMul/ReadVariableOpReadVariableOp+m_2__to__m_3_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_2__to__m_3/MatMul/ReadVariableOp?
m_2__to__m_3/MatMulMatMulm_1__to__m_2/Relu:activations:0*m_2__to__m_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_2__to__m_3/MatMul?
#m_2__to__m_3/BiasAdd/ReadVariableOpReadVariableOp,m_2__to__m_3_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_2__to__m_3/BiasAdd/ReadVariableOp?
m_2__to__m_3/BiasAddAddm_2__to__m_3/MatMul:product:0+m_2__to__m_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_2__to__m_3/BiasAddz
m_2__to__m_3/ReluRelum_2__to__m_3/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_2__to__m_3/Relu?
"m_3__to__m_4/MatMul/ReadVariableOpReadVariableOp+m_3__to__m_4_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_3__to__m_4/MatMul/ReadVariableOp?
m_3__to__m_4/MatMulMatMulm_2__to__m_3/Relu:activations:0*m_3__to__m_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_3__to__m_4/MatMul?
#m_3__to__m_4/BiasAdd/ReadVariableOpReadVariableOp,m_3__to__m_4_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_3__to__m_4/BiasAdd/ReadVariableOp?
m_3__to__m_4/BiasAddAddm_3__to__m_4/MatMul:product:0+m_3__to__m_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_3__to__m_4/BiasAddz
m_3__to__m_4/ReluRelum_3__to__m_4/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_3__to__m_4/Relu?
"m_4__to__m_5/MatMul/ReadVariableOpReadVariableOp+m_4__to__m_5_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_4__to__m_5/MatMul/ReadVariableOp?
m_4__to__m_5/MatMulMatMulm_3__to__m_4/Relu:activations:0*m_4__to__m_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_4__to__m_5/MatMul?
#m_4__to__m_5/BiasAdd/ReadVariableOpReadVariableOp,m_4__to__m_5_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_4__to__m_5/BiasAdd/ReadVariableOp?
m_4__to__m_5/BiasAddAddm_4__to__m_5/MatMul:product:0+m_4__to__m_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_4__to__m_5/BiasAddz
m_4__to__m_5/ReluRelum_4__to__m_5/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_4__to__m_5/Relu?
"m_5__to__m_6/MatMul/ReadVariableOpReadVariableOp+m_5__to__m_6_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_5__to__m_6/MatMul/ReadVariableOp?
m_5__to__m_6/MatMulMatMulm_4__to__m_5/Relu:activations:0*m_5__to__m_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_5__to__m_6/MatMul?
#m_5__to__m_6/BiasAdd/ReadVariableOpReadVariableOp,m_5__to__m_6_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_5__to__m_6/BiasAdd/ReadVariableOp?
m_5__to__m_6/BiasAddAddm_5__to__m_6/MatMul:product:0+m_5__to__m_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_5__to__m_6/BiasAddz
m_5__to__m_6/ReluRelum_5__to__m_6/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_5__to__m_6/Relu?
"m_6__to__m_7/MatMul/ReadVariableOpReadVariableOp+m_6__to__m_7_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_6__to__m_7/MatMul/ReadVariableOp?
m_6__to__m_7/MatMulMatMulm_5__to__m_6/Relu:activations:0*m_6__to__m_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_6__to__m_7/MatMul?
#m_6__to__m_7/BiasAdd/ReadVariableOpReadVariableOp,m_6__to__m_7_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_6__to__m_7/BiasAdd/ReadVariableOp?
m_6__to__m_7/BiasAddAddm_6__to__m_7/MatMul:product:0+m_6__to__m_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_6__to__m_7/BiasAddz
m_6__to__m_7/ReluRelum_6__to__m_7/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_6__to__m_7/Relu?
"m_7__to__m_8/MatMul/ReadVariableOpReadVariableOp+m_7__to__m_8_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_7__to__m_8/MatMul/ReadVariableOp?
m_7__to__m_8/MatMulMatMulm_6__to__m_7/Relu:activations:0*m_7__to__m_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_7__to__m_8/MatMul?
#m_7__to__m_8/BiasAdd/ReadVariableOpReadVariableOp,m_7__to__m_8_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_7__to__m_8/BiasAdd/ReadVariableOp?
m_7__to__m_8/BiasAddAddm_7__to__m_8/MatMul:product:0+m_7__to__m_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_7__to__m_8/BiasAddz
m_7__to__m_8/ReluRelum_7__to__m_8/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_7__to__m_8/Relu?
"m_8__to__m_9/MatMul/ReadVariableOpReadVariableOp+m_8__to__m_9_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_8__to__m_9/MatMul/ReadVariableOp?
m_8__to__m_9/MatMulMatMulm_7__to__m_8/Relu:activations:0*m_8__to__m_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_8__to__m_9/MatMul?
#m_8__to__m_9/BiasAdd/ReadVariableOpReadVariableOp,m_8__to__m_9_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_8__to__m_9/BiasAdd/ReadVariableOp?
m_8__to__m_9/BiasAddAddm_8__to__m_9/MatMul:product:0+m_8__to__m_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_8__to__m_9/BiasAddz
m_8__to__m_9/ReluRelum_8__to__m_9/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_8__to__m_9/Relu?
#m_9__to__m_10/MatMul/ReadVariableOpReadVariableOp,m_9__to__m_10_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02%
#m_9__to__m_10/MatMul/ReadVariableOp?
m_9__to__m_10/MatMulMatMulm_8__to__m_9/Relu:activations:0+m_9__to__m_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_9__to__m_10/MatMul?
$m_9__to__m_10/BiasAdd/ReadVariableOpReadVariableOp-m_9__to__m_10_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02&
$m_9__to__m_10/BiasAdd/ReadVariableOp?
m_9__to__m_10/BiasAddAddm_9__to__m_10/MatMul:product:0,m_9__to__m_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_9__to__m_10/BiasAdd}
m_9__to__m_10/ReluRelum_9__to__m_10/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_9__to__m_10/Relu?
$m_10__to__m_11/MatMul/ReadVariableOpReadVariableOp-m_10__to__m_11_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02&
$m_10__to__m_11/MatMul/ReadVariableOp?
m_10__to__m_11/MatMulMatMul m_9__to__m_10/Relu:activations:0,m_10__to__m_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_10__to__m_11/MatMul?
%m_10__to__m_11/BiasAdd/ReadVariableOpReadVariableOp.m_10__to__m_11_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02'
%m_10__to__m_11/BiasAdd/ReadVariableOp?
m_10__to__m_11/BiasAddAddm_10__to__m_11/MatMul:product:0-m_10__to__m_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_10__to__m_11/BiasAdd?
m_10__to__m_11/ReluRelum_10__to__m_11/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_10__to__m_11/Relu?
$m_11__to__m_12/MatMul/ReadVariableOpReadVariableOp-m_11__to__m_12_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02&
$m_11__to__m_12/MatMul/ReadVariableOp?
m_11__to__m_12/MatMulMatMul!m_10__to__m_11/Relu:activations:0,m_11__to__m_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_11__to__m_12/MatMul?
%m_11__to__m_12/BiasAdd/ReadVariableOpReadVariableOp.m_11__to__m_12_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02'
%m_11__to__m_12/BiasAdd/ReadVariableOp?
m_11__to__m_12/BiasAddAddm_11__to__m_12/MatMul:product:0-m_11__to__m_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_11__to__m_12/BiasAdd?
m_11__to__m_12/ReluRelum_11__to__m_12/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_11__to__m_12/Relu?
$m_12__to__m_13/MatMul/ReadVariableOpReadVariableOp-m_12__to__m_13_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02&
$m_12__to__m_13/MatMul/ReadVariableOp?
m_12__to__m_13/MatMulMatMul!m_11__to__m_12/Relu:activations:0,m_12__to__m_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_12__to__m_13/MatMul?
%m_12__to__m_13/BiasAdd/ReadVariableOpReadVariableOp.m_12__to__m_13_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02'
%m_12__to__m_13/BiasAdd/ReadVariableOp?
m_12__to__m_13/BiasAddAddm_12__to__m_13/MatMul:product:0-m_12__to__m_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_12__to__m_13/BiasAdd?
m_12__to__m_13/ReluRelum_12__to__m_13/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_12__to__m_13/Relu?
$m_13__to__m_14/MatMul/ReadVariableOpReadVariableOp-m_13__to__m_14_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02&
$m_13__to__m_14/MatMul/ReadVariableOp?
m_13__to__m_14/MatMulMatMul!m_12__to__m_13/Relu:activations:0,m_13__to__m_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_13__to__m_14/MatMul?
%m_13__to__m_14/BiasAdd/ReadVariableOpReadVariableOp.m_13__to__m_14_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02'
%m_13__to__m_14/BiasAdd/ReadVariableOp?
m_13__to__m_14/BiasAddAddm_13__to__m_14/MatMul:product:0-m_13__to__m_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_13__to__m_14/BiasAdd?
m_13__to__m_14/ReluRelum_13__to__m_14/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_13__to__m_14/Relu?
$m_14__to__m_15/MatMul/ReadVariableOpReadVariableOp-m_14__to__m_15_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02&
$m_14__to__m_15/MatMul/ReadVariableOp?
m_14__to__m_15/MatMulMatMul!m_13__to__m_14/Relu:activations:0,m_14__to__m_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_14__to__m_15/MatMul?
%m_14__to__m_15/BiasAdd/ReadVariableOpReadVariableOp.m_14__to__m_15_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02'
%m_14__to__m_15/BiasAdd/ReadVariableOp?
m_14__to__m_15/BiasAddAddm_14__to__m_15/MatMul:product:0-m_14__to__m_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_14__to__m_15/BiasAdd?
m_14__to__m_15/ReluRelum_14__to__m_15/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_14__to__m_15/Relu?
$m_15__to__m_16/MatMul/ReadVariableOpReadVariableOp-m_15__to__m_16_matmul_readvariableop_resource*
_output_shapes

:>*
dtype02&
$m_15__to__m_16/MatMul/ReadVariableOp?
m_15__to__m_16/MatMulMatMul!m_14__to__m_15/Relu:activations:0,m_15__to__m_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_15__to__m_16/MatMul?
%m_15__to__m_16/BiasAdd/ReadVariableOpReadVariableOp.m_15__to__m_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%m_15__to__m_16/BiasAdd/ReadVariableOp?
m_15__to__m_16/BiasAddAddm_15__to__m_16/MatMul:product:0-m_15__to__m_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_15__to__m_16/BiasAdd?
m_15__to__m_16/SoftmaxSoftmaxm_15__to__m_16/BiasAdd:z:0*
T0*'
_output_shapes
:?????????2
m_15__to__m_16/Softmax?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:>*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2%
#m_0__to__m_1/kernel/Regularizer/Abs?
%m_0__to__m_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_0__to__m_1/kernel/Regularizer/Const?
#m_0__to__m_1/kernel/Regularizer/SumSum'm_0__to__m_1/kernel/Regularizer/Abs:y:0.m_0__to__m_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/Sum?
%m_0__to__m_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_0__to__m_1/kernel/Regularizer/mul/x?
#m_0__to__m_1/kernel/Regularizer/mulMul.m_0__to__m_1/kernel/Regularizer/mul/x:output:0,m_0__to__m_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/mul?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_1__to__m_2_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_1__to__m_2/kernel/Regularizer/Abs?
%m_1__to__m_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_1__to__m_2/kernel/Regularizer/Const?
#m_1__to__m_2/kernel/Regularizer/SumSum'm_1__to__m_2/kernel/Regularizer/Abs:y:0.m_1__to__m_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/Sum?
%m_1__to__m_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_1__to__m_2/kernel/Regularizer/mul/x?
#m_1__to__m_2/kernel/Regularizer/mulMul.m_1__to__m_2/kernel/Regularizer/mul/x:output:0,m_1__to__m_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/mul?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_2__to__m_3_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_2__to__m_3/kernel/Regularizer/Abs?
%m_2__to__m_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_2__to__m_3/kernel/Regularizer/Const?
#m_2__to__m_3/kernel/Regularizer/SumSum'm_2__to__m_3/kernel/Regularizer/Abs:y:0.m_2__to__m_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/Sum?
%m_2__to__m_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_2__to__m_3/kernel/Regularizer/mul/x?
#m_2__to__m_3/kernel/Regularizer/mulMul.m_2__to__m_3/kernel/Regularizer/mul/x:output:0,m_2__to__m_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/mul?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_3__to__m_4_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_3__to__m_4/kernel/Regularizer/Abs?
%m_3__to__m_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_3__to__m_4/kernel/Regularizer/Const?
#m_3__to__m_4/kernel/Regularizer/SumSum'm_3__to__m_4/kernel/Regularizer/Abs:y:0.m_3__to__m_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/Sum?
%m_3__to__m_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_3__to__m_4/kernel/Regularizer/mul/x?
#m_3__to__m_4/kernel/Regularizer/mulMul.m_3__to__m_4/kernel/Regularizer/mul/x:output:0,m_3__to__m_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/mul?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_4__to__m_5_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_4__to__m_5/kernel/Regularizer/Abs?
%m_4__to__m_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_4__to__m_5/kernel/Regularizer/Const?
#m_4__to__m_5/kernel/Regularizer/SumSum'm_4__to__m_5/kernel/Regularizer/Abs:y:0.m_4__to__m_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/Sum?
%m_4__to__m_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_4__to__m_5/kernel/Regularizer/mul/x?
#m_4__to__m_5/kernel/Regularizer/mulMul.m_4__to__m_5/kernel/Regularizer/mul/x:output:0,m_4__to__m_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/mul?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_5__to__m_6_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_5__to__m_6/kernel/Regularizer/Abs?
%m_5__to__m_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_5__to__m_6/kernel/Regularizer/Const?
#m_5__to__m_6/kernel/Regularizer/SumSum'm_5__to__m_6/kernel/Regularizer/Abs:y:0.m_5__to__m_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/Sum?
%m_5__to__m_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_5__to__m_6/kernel/Regularizer/mul/x?
#m_5__to__m_6/kernel/Regularizer/mulMul.m_5__to__m_6/kernel/Regularizer/mul/x:output:0,m_5__to__m_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/mul?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_6__to__m_7_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_6__to__m_7/kernel/Regularizer/Abs?
%m_6__to__m_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_6__to__m_7/kernel/Regularizer/Const?
#m_6__to__m_7/kernel/Regularizer/SumSum'm_6__to__m_7/kernel/Regularizer/Abs:y:0.m_6__to__m_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/Sum?
%m_6__to__m_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_6__to__m_7/kernel/Regularizer/mul/x?
#m_6__to__m_7/kernel/Regularizer/mulMul.m_6__to__m_7/kernel/Regularizer/mul/x:output:0,m_6__to__m_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/mul?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_7__to__m_8_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_7__to__m_8/kernel/Regularizer/Abs?
%m_7__to__m_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_7__to__m_8/kernel/Regularizer/Const?
#m_7__to__m_8/kernel/Regularizer/SumSum'm_7__to__m_8/kernel/Regularizer/Abs:y:0.m_7__to__m_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/Sum?
%m_7__to__m_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_7__to__m_8/kernel/Regularizer/mul/x?
#m_7__to__m_8/kernel/Regularizer/mulMul.m_7__to__m_8/kernel/Regularizer/mul/x:output:0,m_7__to__m_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/mul?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_8__to__m_9_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_8__to__m_9/kernel/Regularizer/Abs?
%m_8__to__m_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_8__to__m_9/kernel/Regularizer/Const?
#m_8__to__m_9/kernel/Regularizer/SumSum'm_8__to__m_9/kernel/Regularizer/Abs:y:0.m_8__to__m_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/Sum?
%m_8__to__m_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_8__to__m_9/kernel/Regularizer/mul/x?
#m_8__to__m_9/kernel/Regularizer/mulMul.m_8__to__m_9/kernel/Regularizer/mul/x:output:0,m_8__to__m_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/mul?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp,m_9__to__m_10_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2&
$m_9__to__m_10/kernel/Regularizer/Abs?
&m_9__to__m_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&m_9__to__m_10/kernel/Regularizer/Const?
$m_9__to__m_10/kernel/Regularizer/SumSum(m_9__to__m_10/kernel/Regularizer/Abs:y:0/m_9__to__m_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/Sum?
&m_9__to__m_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92(
&m_9__to__m_10/kernel/Regularizer/mul/x?
$m_9__to__m_10/kernel/Regularizer/mulMul/m_9__to__m_10/kernel/Regularizer/mul/x:output:0-m_9__to__m_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/mul?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_10__to__m_11_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_10__to__m_11/kernel/Regularizer/Abs?
'm_10__to__m_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_10__to__m_11/kernel/Regularizer/Const?
%m_10__to__m_11/kernel/Regularizer/SumSum)m_10__to__m_11/kernel/Regularizer/Abs:y:00m_10__to__m_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/Sum?
'm_10__to__m_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_10__to__m_11/kernel/Regularizer/mul/x?
%m_10__to__m_11/kernel/Regularizer/mulMul0m_10__to__m_11/kernel/Regularizer/mul/x:output:0.m_10__to__m_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/mul?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_11__to__m_12_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_11__to__m_12/kernel/Regularizer/Abs?
'm_11__to__m_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_11__to__m_12/kernel/Regularizer/Const?
%m_11__to__m_12/kernel/Regularizer/SumSum)m_11__to__m_12/kernel/Regularizer/Abs:y:00m_11__to__m_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/Sum?
'm_11__to__m_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_11__to__m_12/kernel/Regularizer/mul/x?
%m_11__to__m_12/kernel/Regularizer/mulMul0m_11__to__m_12/kernel/Regularizer/mul/x:output:0.m_11__to__m_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/mul?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_12__to__m_13_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_12__to__m_13/kernel/Regularizer/Abs?
'm_12__to__m_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_12__to__m_13/kernel/Regularizer/Const?
%m_12__to__m_13/kernel/Regularizer/SumSum)m_12__to__m_13/kernel/Regularizer/Abs:y:00m_12__to__m_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/Sum?
'm_12__to__m_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_12__to__m_13/kernel/Regularizer/mul/x?
%m_12__to__m_13/kernel/Regularizer/mulMul0m_12__to__m_13/kernel/Regularizer/mul/x:output:0.m_12__to__m_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/mul?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_13__to__m_14_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_13__to__m_14/kernel/Regularizer/Abs?
'm_13__to__m_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_13__to__m_14/kernel/Regularizer/Const?
%m_13__to__m_14/kernel/Regularizer/SumSum)m_13__to__m_14/kernel/Regularizer/Abs:y:00m_13__to__m_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/Sum?
'm_13__to__m_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_13__to__m_14/kernel/Regularizer/mul/x?
%m_13__to__m_14/kernel/Regularizer/mulMul0m_13__to__m_14/kernel/Regularizer/mul/x:output:0.m_13__to__m_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/mul?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_14__to__m_15_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_14__to__m_15/kernel/Regularizer/Abs?
'm_14__to__m_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_14__to__m_15/kernel/Regularizer/Const?
%m_14__to__m_15/kernel/Regularizer/SumSum)m_14__to__m_15/kernel/Regularizer/Abs:y:00m_14__to__m_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/Sum?
'm_14__to__m_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_14__to__m_15/kernel/Regularizer/mul/x?
%m_14__to__m_15/kernel/Regularizer/mulMul0m_14__to__m_15/kernel/Regularizer/mul/x:output:0.m_14__to__m_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/mul?
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_15__to__m_16_matmul_readvariableop_resource*
_output_shapes

:>*
dtype026
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
%m_15__to__m_16/kernel/Regularizer/AbsAbs<m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2'
%m_15__to__m_16/kernel/Regularizer/Abs?
'm_15__to__m_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_15__to__m_16/kernel/Regularizer/Const?
%m_15__to__m_16/kernel/Regularizer/SumSum)m_15__to__m_16/kernel/Regularizer/Abs:y:00m_15__to__m_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/Sum?
'm_15__to__m_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_15__to__m_16/kernel/Regularizer/mul/x?
%m_15__to__m_16/kernel/Regularizer/mulMul0m_15__to__m_16/kernel/Regularizer/mul/x:output:0.m_15__to__m_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/mul?
IdentityIdentity m_15__to__m_16/Softmax:softmax:0$^m_0__to__m_1/BiasAdd/ReadVariableOp#^m_0__to__m_1/MatMul/ReadVariableOp3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp&^m_10__to__m_11/BiasAdd/ReadVariableOp%^m_10__to__m_11/MatMul/ReadVariableOp5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp&^m_11__to__m_12/BiasAdd/ReadVariableOp%^m_11__to__m_12/MatMul/ReadVariableOp5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp&^m_12__to__m_13/BiasAdd/ReadVariableOp%^m_12__to__m_13/MatMul/ReadVariableOp5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp&^m_13__to__m_14/BiasAdd/ReadVariableOp%^m_13__to__m_14/MatMul/ReadVariableOp5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp&^m_14__to__m_15/BiasAdd/ReadVariableOp%^m_14__to__m_15/MatMul/ReadVariableOp5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp&^m_15__to__m_16/BiasAdd/ReadVariableOp%^m_15__to__m_16/MatMul/ReadVariableOp5^m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp$^m_1__to__m_2/BiasAdd/ReadVariableOp#^m_1__to__m_2/MatMul/ReadVariableOp3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp$^m_2__to__m_3/BiasAdd/ReadVariableOp#^m_2__to__m_3/MatMul/ReadVariableOp3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp$^m_3__to__m_4/BiasAdd/ReadVariableOp#^m_3__to__m_4/MatMul/ReadVariableOp3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp$^m_4__to__m_5/BiasAdd/ReadVariableOp#^m_4__to__m_5/MatMul/ReadVariableOp3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp$^m_5__to__m_6/BiasAdd/ReadVariableOp#^m_5__to__m_6/MatMul/ReadVariableOp3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp$^m_6__to__m_7/BiasAdd/ReadVariableOp#^m_6__to__m_7/MatMul/ReadVariableOp3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp$^m_7__to__m_8/BiasAdd/ReadVariableOp#^m_7__to__m_8/MatMul/ReadVariableOp3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp$^m_8__to__m_9/BiasAdd/ReadVariableOp#^m_8__to__m_9/MatMul/ReadVariableOp3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp%^m_9__to__m_10/BiasAdd/ReadVariableOp$^m_9__to__m_10/MatMul/ReadVariableOp4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#m_0__to__m_1/BiasAdd/ReadVariableOp#m_0__to__m_1/BiasAdd/ReadVariableOp2H
"m_0__to__m_1/MatMul/ReadVariableOp"m_0__to__m_1/MatMul/ReadVariableOp2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2N
%m_10__to__m_11/BiasAdd/ReadVariableOp%m_10__to__m_11/BiasAdd/ReadVariableOp2L
$m_10__to__m_11/MatMul/ReadVariableOp$m_10__to__m_11/MatMul/ReadVariableOp2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp2N
%m_11__to__m_12/BiasAdd/ReadVariableOp%m_11__to__m_12/BiasAdd/ReadVariableOp2L
$m_11__to__m_12/MatMul/ReadVariableOp$m_11__to__m_12/MatMul/ReadVariableOp2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp2N
%m_12__to__m_13/BiasAdd/ReadVariableOp%m_12__to__m_13/BiasAdd/ReadVariableOp2L
$m_12__to__m_13/MatMul/ReadVariableOp$m_12__to__m_13/MatMul/ReadVariableOp2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp2N
%m_13__to__m_14/BiasAdd/ReadVariableOp%m_13__to__m_14/BiasAdd/ReadVariableOp2L
$m_13__to__m_14/MatMul/ReadVariableOp$m_13__to__m_14/MatMul/ReadVariableOp2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp2N
%m_14__to__m_15/BiasAdd/ReadVariableOp%m_14__to__m_15/BiasAdd/ReadVariableOp2L
$m_14__to__m_15/MatMul/ReadVariableOp$m_14__to__m_15/MatMul/ReadVariableOp2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2N
%m_15__to__m_16/BiasAdd/ReadVariableOp%m_15__to__m_16/BiasAdd/ReadVariableOp2L
$m_15__to__m_16/MatMul/ReadVariableOp$m_15__to__m_16/MatMul/ReadVariableOp2l
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp2J
#m_1__to__m_2/BiasAdd/ReadVariableOp#m_1__to__m_2/BiasAdd/ReadVariableOp2H
"m_1__to__m_2/MatMul/ReadVariableOp"m_1__to__m_2/MatMul/ReadVariableOp2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2J
#m_2__to__m_3/BiasAdd/ReadVariableOp#m_2__to__m_3/BiasAdd/ReadVariableOp2H
"m_2__to__m_3/MatMul/ReadVariableOp"m_2__to__m_3/MatMul/ReadVariableOp2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2J
#m_3__to__m_4/BiasAdd/ReadVariableOp#m_3__to__m_4/BiasAdd/ReadVariableOp2H
"m_3__to__m_4/MatMul/ReadVariableOp"m_3__to__m_4/MatMul/ReadVariableOp2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2J
#m_4__to__m_5/BiasAdd/ReadVariableOp#m_4__to__m_5/BiasAdd/ReadVariableOp2H
"m_4__to__m_5/MatMul/ReadVariableOp"m_4__to__m_5/MatMul/ReadVariableOp2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2J
#m_5__to__m_6/BiasAdd/ReadVariableOp#m_5__to__m_6/BiasAdd/ReadVariableOp2H
"m_5__to__m_6/MatMul/ReadVariableOp"m_5__to__m_6/MatMul/ReadVariableOp2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2J
#m_6__to__m_7/BiasAdd/ReadVariableOp#m_6__to__m_7/BiasAdd/ReadVariableOp2H
"m_6__to__m_7/MatMul/ReadVariableOp"m_6__to__m_7/MatMul/ReadVariableOp2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2J
#m_7__to__m_8/BiasAdd/ReadVariableOp#m_7__to__m_8/BiasAdd/ReadVariableOp2H
"m_7__to__m_8/MatMul/ReadVariableOp"m_7__to__m_8/MatMul/ReadVariableOp2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2J
#m_8__to__m_9/BiasAdd/ReadVariableOp#m_8__to__m_9/BiasAdd/ReadVariableOp2H
"m_8__to__m_9/MatMul/ReadVariableOp"m_8__to__m_9/MatMul/ReadVariableOp2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2L
$m_9__to__m_10/BiasAdd/ReadVariableOp$m_9__to__m_10/BiasAdd/ReadVariableOp2J
#m_9__to__m_10/MatMul/ReadVariableOp#m_9__to__m_10/MatMul/ReadVariableOp2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_13989763

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_10__to__m_11/kernel/Regularizer/Abs?
'm_10__to__m_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_10__to__m_11/kernel/Regularizer/Const?
%m_10__to__m_11/kernel/Regularizer/SumSum)m_10__to__m_11/kernel/Regularizer/Abs:y:00m_10__to__m_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/Sum?
'm_10__to__m_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_10__to__m_11/kernel/Regularizer/mul/x?
%m_10__to__m_11/kernel/Regularizer/mulMul0m_10__to__m_11/kernel/Regularizer/mul/x:output:0.m_10__to__m_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_13991128
m_0__to__m_1_input
unknown:>
	unknown_0:>
	unknown_1:>>
	unknown_2:>
	unknown_3:>>
	unknown_4:>
	unknown_5:>>
	unknown_6:>
	unknown_7:>>
	unknown_8:>
	unknown_9:>>

unknown_10:>

unknown_11:>>

unknown_12:>

unknown_13:>>

unknown_14:>

unknown_15:>>

unknown_16:>

unknown_17:>>

unknown_18:>

unknown_19:>>

unknown_20:>

unknown_21:>>

unknown_22:>

unknown_23:>>

unknown_24:>

unknown_25:>>

unknown_26:>

unknown_27:>>

unknown_28:>

unknown_29:>

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallm_0__to__m_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_139895092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_13989740

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2&
$m_9__to__m_10/kernel/Regularizer/Abs?
&m_9__to__m_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&m_9__to__m_10/kernel/Regularizer/Const?
$m_9__to__m_10/kernel/Regularizer/SumSum(m_9__to__m_10/kernel/Regularizer/Abs:y:0/m_9__to__m_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/Sum?
&m_9__to__m_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92(
&m_9__to__m_10/kernel/Regularizer/mul/x?
$m_9__to__m_10/kernel/Regularizer/mulMul/m_9__to__m_10/kernel/Regularizer/mul/x:output:0-m_9__to__m_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
__inference_loss_fn_14_13992367O
=m_14__to__m_15_kernel_regularizer_abs_readvariableop_resource:>>
identity??4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_14__to__m_15_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_14__to__m_15/kernel/Regularizer/Abs?
'm_14__to__m_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_14__to__m_15/kernel/Regularizer/Const?
%m_14__to__m_15/kernel/Regularizer/SumSum)m_14__to__m_15/kernel/Regularizer/Abs:y:00m_14__to__m_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/Sum?
'm_14__to__m_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_14__to__m_15/kernel/Regularizer/mul/x?
%m_14__to__m_15/kernel/Regularizer/mulMul0m_14__to__m_15/kernel/Regularizer/mul/x:output:0.m_14__to__m_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/mul?
IdentityIdentity)m_14__to__m_15/kernel/Regularizer/mul:z:05^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp
?
?
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_13989625

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_4__to__m_5/kernel/Regularizer/Abs?
%m_4__to__m_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_4__to__m_5/kernel/Regularizer/Const?
#m_4__to__m_5/kernel/Regularizer/SumSum'm_4__to__m_5/kernel/Regularizer/Abs:y:0.m_4__to__m_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/Sum?
%m_4__to__m_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_4__to__m_5/kernel/Regularizer/mul/x?
#m_4__to__m_5/kernel/Regularizer/mulMul.m_4__to__m_5/kernel/Regularizer/mul/x:output:0,m_4__to__m_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
L__inference_m_15__to__m_16_layer_call_and_return_conditional_losses_13989878

inputs0
matmul_readvariableop_resource:>-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd\
SoftmaxSoftmaxBiasAdd:z:0*
T0*'
_output_shapes
:?????????2	
Softmax?
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>*
dtype026
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
%m_15__to__m_16/kernel/Regularizer/AbsAbs<m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2'
%m_15__to__m_16/kernel/Regularizer/Abs?
'm_15__to__m_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_15__to__m_16/kernel/Regularizer/Const?
%m_15__to__m_16/kernel/Regularizer/SumSum)m_15__to__m_16/kernel/Regularizer/Abs:y:00m_15__to__m_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/Sum?
'm_15__to__m_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_15__to__m_16/kernel/Regularizer/mul/x?
%m_15__to__m_16/kernel/Regularizer/mulMul0m_15__to__m_16/kernel/Regularizer/mul/x:output:0.m_15__to__m_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/mul?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
??
?
#__inference__wrapped_model_13989509
m_0__to__m_1_inputH
6l_15_m_0_1_m_0__to__m_1_matmul_readvariableop_resource:>E
7l_15_m_0_1_m_0__to__m_1_biasadd_readvariableop_resource:>H
6l_15_m_0_1_m_1__to__m_2_matmul_readvariableop_resource:>>E
7l_15_m_0_1_m_1__to__m_2_biasadd_readvariableop_resource:>H
6l_15_m_0_1_m_2__to__m_3_matmul_readvariableop_resource:>>E
7l_15_m_0_1_m_2__to__m_3_biasadd_readvariableop_resource:>H
6l_15_m_0_1_m_3__to__m_4_matmul_readvariableop_resource:>>E
7l_15_m_0_1_m_3__to__m_4_biasadd_readvariableop_resource:>H
6l_15_m_0_1_m_4__to__m_5_matmul_readvariableop_resource:>>E
7l_15_m_0_1_m_4__to__m_5_biasadd_readvariableop_resource:>H
6l_15_m_0_1_m_5__to__m_6_matmul_readvariableop_resource:>>E
7l_15_m_0_1_m_5__to__m_6_biasadd_readvariableop_resource:>H
6l_15_m_0_1_m_6__to__m_7_matmul_readvariableop_resource:>>E
7l_15_m_0_1_m_6__to__m_7_biasadd_readvariableop_resource:>H
6l_15_m_0_1_m_7__to__m_8_matmul_readvariableop_resource:>>E
7l_15_m_0_1_m_7__to__m_8_biasadd_readvariableop_resource:>H
6l_15_m_0_1_m_8__to__m_9_matmul_readvariableop_resource:>>E
7l_15_m_0_1_m_8__to__m_9_biasadd_readvariableop_resource:>I
7l_15_m_0_1_m_9__to__m_10_matmul_readvariableop_resource:>>F
8l_15_m_0_1_m_9__to__m_10_biasadd_readvariableop_resource:>J
8l_15_m_0_1_m_10__to__m_11_matmul_readvariableop_resource:>>G
9l_15_m_0_1_m_10__to__m_11_biasadd_readvariableop_resource:>J
8l_15_m_0_1_m_11__to__m_12_matmul_readvariableop_resource:>>G
9l_15_m_0_1_m_11__to__m_12_biasadd_readvariableop_resource:>J
8l_15_m_0_1_m_12__to__m_13_matmul_readvariableop_resource:>>G
9l_15_m_0_1_m_12__to__m_13_biasadd_readvariableop_resource:>J
8l_15_m_0_1_m_13__to__m_14_matmul_readvariableop_resource:>>G
9l_15_m_0_1_m_13__to__m_14_biasadd_readvariableop_resource:>J
8l_15_m_0_1_m_14__to__m_15_matmul_readvariableop_resource:>>G
9l_15_m_0_1_m_14__to__m_15_biasadd_readvariableop_resource:>J
8l_15_m_0_1_m_15__to__m_16_matmul_readvariableop_resource:>G
9l_15_m_0_1_m_15__to__m_16_biasadd_readvariableop_resource:
identity??.L-15-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp?-L-15-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp?0L-15-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp?/L-15-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp?0L-15-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp?/L-15-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp?0L-15-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp?/L-15-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp?0L-15-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp?/L-15-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp?0L-15-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp?/L-15-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp?0L-15-m_0-1/m_15__to__m_16/BiasAdd/ReadVariableOp?/L-15-m_0-1/m_15__to__m_16/MatMul/ReadVariableOp?.L-15-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp?-L-15-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp?.L-15-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp?-L-15-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp?.L-15-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp?-L-15-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp?.L-15-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp?-L-15-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp?.L-15-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp?-L-15-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp?.L-15-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp?-L-15-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp?.L-15-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp?-L-15-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp?.L-15-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp?-L-15-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp?/L-15-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp?.L-15-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp?
-L-15-m_0-1/m_0__to__m_1/MatMul/ReadVariableOpReadVariableOp6l_15_m_0_1_m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:>*
dtype02/
-L-15-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp?
L-15-m_0-1/m_0__to__m_1/MatMulMatMulm_0__to__m_1_input5L-15-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_0__to__m_1/MatMul?
.L-15-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOpReadVariableOp7l_15_m_0_1_m_0__to__m_1_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype020
.L-15-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp?
L-15-m_0-1/m_0__to__m_1/BiasAddAdd(L-15-m_0-1/m_0__to__m_1/MatMul:product:06L-15-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2!
L-15-m_0-1/m_0__to__m_1/BiasAdd?
L-15-m_0-1/m_0__to__m_1/ReluRelu#L-15-m_0-1/m_0__to__m_1/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
L-15-m_0-1/m_0__to__m_1/Relu?
-L-15-m_0-1/m_1__to__m_2/MatMul/ReadVariableOpReadVariableOp6l_15_m_0_1_m_1__to__m_2_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02/
-L-15-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp?
L-15-m_0-1/m_1__to__m_2/MatMulMatMul*L-15-m_0-1/m_0__to__m_1/Relu:activations:05L-15-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_1__to__m_2/MatMul?
.L-15-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOpReadVariableOp7l_15_m_0_1_m_1__to__m_2_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype020
.L-15-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp?
L-15-m_0-1/m_1__to__m_2/BiasAddAdd(L-15-m_0-1/m_1__to__m_2/MatMul:product:06L-15-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2!
L-15-m_0-1/m_1__to__m_2/BiasAdd?
L-15-m_0-1/m_1__to__m_2/ReluRelu#L-15-m_0-1/m_1__to__m_2/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
L-15-m_0-1/m_1__to__m_2/Relu?
-L-15-m_0-1/m_2__to__m_3/MatMul/ReadVariableOpReadVariableOp6l_15_m_0_1_m_2__to__m_3_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02/
-L-15-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp?
L-15-m_0-1/m_2__to__m_3/MatMulMatMul*L-15-m_0-1/m_1__to__m_2/Relu:activations:05L-15-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_2__to__m_3/MatMul?
.L-15-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOpReadVariableOp7l_15_m_0_1_m_2__to__m_3_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype020
.L-15-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp?
L-15-m_0-1/m_2__to__m_3/BiasAddAdd(L-15-m_0-1/m_2__to__m_3/MatMul:product:06L-15-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2!
L-15-m_0-1/m_2__to__m_3/BiasAdd?
L-15-m_0-1/m_2__to__m_3/ReluRelu#L-15-m_0-1/m_2__to__m_3/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
L-15-m_0-1/m_2__to__m_3/Relu?
-L-15-m_0-1/m_3__to__m_4/MatMul/ReadVariableOpReadVariableOp6l_15_m_0_1_m_3__to__m_4_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02/
-L-15-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp?
L-15-m_0-1/m_3__to__m_4/MatMulMatMul*L-15-m_0-1/m_2__to__m_3/Relu:activations:05L-15-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_3__to__m_4/MatMul?
.L-15-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOpReadVariableOp7l_15_m_0_1_m_3__to__m_4_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype020
.L-15-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp?
L-15-m_0-1/m_3__to__m_4/BiasAddAdd(L-15-m_0-1/m_3__to__m_4/MatMul:product:06L-15-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2!
L-15-m_0-1/m_3__to__m_4/BiasAdd?
L-15-m_0-1/m_3__to__m_4/ReluRelu#L-15-m_0-1/m_3__to__m_4/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
L-15-m_0-1/m_3__to__m_4/Relu?
-L-15-m_0-1/m_4__to__m_5/MatMul/ReadVariableOpReadVariableOp6l_15_m_0_1_m_4__to__m_5_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02/
-L-15-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp?
L-15-m_0-1/m_4__to__m_5/MatMulMatMul*L-15-m_0-1/m_3__to__m_4/Relu:activations:05L-15-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_4__to__m_5/MatMul?
.L-15-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOpReadVariableOp7l_15_m_0_1_m_4__to__m_5_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype020
.L-15-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp?
L-15-m_0-1/m_4__to__m_5/BiasAddAdd(L-15-m_0-1/m_4__to__m_5/MatMul:product:06L-15-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2!
L-15-m_0-1/m_4__to__m_5/BiasAdd?
L-15-m_0-1/m_4__to__m_5/ReluRelu#L-15-m_0-1/m_4__to__m_5/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
L-15-m_0-1/m_4__to__m_5/Relu?
-L-15-m_0-1/m_5__to__m_6/MatMul/ReadVariableOpReadVariableOp6l_15_m_0_1_m_5__to__m_6_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02/
-L-15-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp?
L-15-m_0-1/m_5__to__m_6/MatMulMatMul*L-15-m_0-1/m_4__to__m_5/Relu:activations:05L-15-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_5__to__m_6/MatMul?
.L-15-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOpReadVariableOp7l_15_m_0_1_m_5__to__m_6_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype020
.L-15-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp?
L-15-m_0-1/m_5__to__m_6/BiasAddAdd(L-15-m_0-1/m_5__to__m_6/MatMul:product:06L-15-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2!
L-15-m_0-1/m_5__to__m_6/BiasAdd?
L-15-m_0-1/m_5__to__m_6/ReluRelu#L-15-m_0-1/m_5__to__m_6/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
L-15-m_0-1/m_5__to__m_6/Relu?
-L-15-m_0-1/m_6__to__m_7/MatMul/ReadVariableOpReadVariableOp6l_15_m_0_1_m_6__to__m_7_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02/
-L-15-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp?
L-15-m_0-1/m_6__to__m_7/MatMulMatMul*L-15-m_0-1/m_5__to__m_6/Relu:activations:05L-15-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_6__to__m_7/MatMul?
.L-15-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOpReadVariableOp7l_15_m_0_1_m_6__to__m_7_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype020
.L-15-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp?
L-15-m_0-1/m_6__to__m_7/BiasAddAdd(L-15-m_0-1/m_6__to__m_7/MatMul:product:06L-15-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2!
L-15-m_0-1/m_6__to__m_7/BiasAdd?
L-15-m_0-1/m_6__to__m_7/ReluRelu#L-15-m_0-1/m_6__to__m_7/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
L-15-m_0-1/m_6__to__m_7/Relu?
-L-15-m_0-1/m_7__to__m_8/MatMul/ReadVariableOpReadVariableOp6l_15_m_0_1_m_7__to__m_8_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02/
-L-15-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp?
L-15-m_0-1/m_7__to__m_8/MatMulMatMul*L-15-m_0-1/m_6__to__m_7/Relu:activations:05L-15-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_7__to__m_8/MatMul?
.L-15-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOpReadVariableOp7l_15_m_0_1_m_7__to__m_8_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype020
.L-15-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp?
L-15-m_0-1/m_7__to__m_8/BiasAddAdd(L-15-m_0-1/m_7__to__m_8/MatMul:product:06L-15-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2!
L-15-m_0-1/m_7__to__m_8/BiasAdd?
L-15-m_0-1/m_7__to__m_8/ReluRelu#L-15-m_0-1/m_7__to__m_8/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
L-15-m_0-1/m_7__to__m_8/Relu?
-L-15-m_0-1/m_8__to__m_9/MatMul/ReadVariableOpReadVariableOp6l_15_m_0_1_m_8__to__m_9_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02/
-L-15-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp?
L-15-m_0-1/m_8__to__m_9/MatMulMatMul*L-15-m_0-1/m_7__to__m_8/Relu:activations:05L-15-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_8__to__m_9/MatMul?
.L-15-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOpReadVariableOp7l_15_m_0_1_m_8__to__m_9_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype020
.L-15-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp?
L-15-m_0-1/m_8__to__m_9/BiasAddAdd(L-15-m_0-1/m_8__to__m_9/MatMul:product:06L-15-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2!
L-15-m_0-1/m_8__to__m_9/BiasAdd?
L-15-m_0-1/m_8__to__m_9/ReluRelu#L-15-m_0-1/m_8__to__m_9/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
L-15-m_0-1/m_8__to__m_9/Relu?
.L-15-m_0-1/m_9__to__m_10/MatMul/ReadVariableOpReadVariableOp7l_15_m_0_1_m_9__to__m_10_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype020
.L-15-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp?
L-15-m_0-1/m_9__to__m_10/MatMulMatMul*L-15-m_0-1/m_8__to__m_9/Relu:activations:06L-15-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2!
L-15-m_0-1/m_9__to__m_10/MatMul?
/L-15-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOpReadVariableOp8l_15_m_0_1_m_9__to__m_10_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype021
/L-15-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp?
 L-15-m_0-1/m_9__to__m_10/BiasAddAdd)L-15-m_0-1/m_9__to__m_10/MatMul:product:07L-15-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2"
 L-15-m_0-1/m_9__to__m_10/BiasAdd?
L-15-m_0-1/m_9__to__m_10/ReluRelu$L-15-m_0-1/m_9__to__m_10/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
L-15-m_0-1/m_9__to__m_10/Relu?
/L-15-m_0-1/m_10__to__m_11/MatMul/ReadVariableOpReadVariableOp8l_15_m_0_1_m_10__to__m_11_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype021
/L-15-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp?
 L-15-m_0-1/m_10__to__m_11/MatMulMatMul+L-15-m_0-1/m_9__to__m_10/Relu:activations:07L-15-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2"
 L-15-m_0-1/m_10__to__m_11/MatMul?
0L-15-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOpReadVariableOp9l_15_m_0_1_m_10__to__m_11_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype022
0L-15-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp?
!L-15-m_0-1/m_10__to__m_11/BiasAddAdd*L-15-m_0-1/m_10__to__m_11/MatMul:product:08L-15-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2#
!L-15-m_0-1/m_10__to__m_11/BiasAdd?
L-15-m_0-1/m_10__to__m_11/ReluRelu%L-15-m_0-1/m_10__to__m_11/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_10__to__m_11/Relu?
/L-15-m_0-1/m_11__to__m_12/MatMul/ReadVariableOpReadVariableOp8l_15_m_0_1_m_11__to__m_12_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype021
/L-15-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp?
 L-15-m_0-1/m_11__to__m_12/MatMulMatMul,L-15-m_0-1/m_10__to__m_11/Relu:activations:07L-15-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2"
 L-15-m_0-1/m_11__to__m_12/MatMul?
0L-15-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOpReadVariableOp9l_15_m_0_1_m_11__to__m_12_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype022
0L-15-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp?
!L-15-m_0-1/m_11__to__m_12/BiasAddAdd*L-15-m_0-1/m_11__to__m_12/MatMul:product:08L-15-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2#
!L-15-m_0-1/m_11__to__m_12/BiasAdd?
L-15-m_0-1/m_11__to__m_12/ReluRelu%L-15-m_0-1/m_11__to__m_12/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_11__to__m_12/Relu?
/L-15-m_0-1/m_12__to__m_13/MatMul/ReadVariableOpReadVariableOp8l_15_m_0_1_m_12__to__m_13_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype021
/L-15-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp?
 L-15-m_0-1/m_12__to__m_13/MatMulMatMul,L-15-m_0-1/m_11__to__m_12/Relu:activations:07L-15-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2"
 L-15-m_0-1/m_12__to__m_13/MatMul?
0L-15-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOpReadVariableOp9l_15_m_0_1_m_12__to__m_13_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype022
0L-15-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp?
!L-15-m_0-1/m_12__to__m_13/BiasAddAdd*L-15-m_0-1/m_12__to__m_13/MatMul:product:08L-15-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2#
!L-15-m_0-1/m_12__to__m_13/BiasAdd?
L-15-m_0-1/m_12__to__m_13/ReluRelu%L-15-m_0-1/m_12__to__m_13/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_12__to__m_13/Relu?
/L-15-m_0-1/m_13__to__m_14/MatMul/ReadVariableOpReadVariableOp8l_15_m_0_1_m_13__to__m_14_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype021
/L-15-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp?
 L-15-m_0-1/m_13__to__m_14/MatMulMatMul,L-15-m_0-1/m_12__to__m_13/Relu:activations:07L-15-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2"
 L-15-m_0-1/m_13__to__m_14/MatMul?
0L-15-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOpReadVariableOp9l_15_m_0_1_m_13__to__m_14_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype022
0L-15-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp?
!L-15-m_0-1/m_13__to__m_14/BiasAddAdd*L-15-m_0-1/m_13__to__m_14/MatMul:product:08L-15-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2#
!L-15-m_0-1/m_13__to__m_14/BiasAdd?
L-15-m_0-1/m_13__to__m_14/ReluRelu%L-15-m_0-1/m_13__to__m_14/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_13__to__m_14/Relu?
/L-15-m_0-1/m_14__to__m_15/MatMul/ReadVariableOpReadVariableOp8l_15_m_0_1_m_14__to__m_15_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype021
/L-15-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp?
 L-15-m_0-1/m_14__to__m_15/MatMulMatMul,L-15-m_0-1/m_13__to__m_14/Relu:activations:07L-15-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2"
 L-15-m_0-1/m_14__to__m_15/MatMul?
0L-15-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOpReadVariableOp9l_15_m_0_1_m_14__to__m_15_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype022
0L-15-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp?
!L-15-m_0-1/m_14__to__m_15/BiasAddAdd*L-15-m_0-1/m_14__to__m_15/MatMul:product:08L-15-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2#
!L-15-m_0-1/m_14__to__m_15/BiasAdd?
L-15-m_0-1/m_14__to__m_15/ReluRelu%L-15-m_0-1/m_14__to__m_15/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2 
L-15-m_0-1/m_14__to__m_15/Relu?
/L-15-m_0-1/m_15__to__m_16/MatMul/ReadVariableOpReadVariableOp8l_15_m_0_1_m_15__to__m_16_matmul_readvariableop_resource*
_output_shapes

:>*
dtype021
/L-15-m_0-1/m_15__to__m_16/MatMul/ReadVariableOp?
 L-15-m_0-1/m_15__to__m_16/MatMulMatMul,L-15-m_0-1/m_14__to__m_15/Relu:activations:07L-15-m_0-1/m_15__to__m_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 L-15-m_0-1/m_15__to__m_16/MatMul?
0L-15-m_0-1/m_15__to__m_16/BiasAdd/ReadVariableOpReadVariableOp9l_15_m_0_1_m_15__to__m_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0L-15-m_0-1/m_15__to__m_16/BiasAdd/ReadVariableOp?
!L-15-m_0-1/m_15__to__m_16/BiasAddAdd*L-15-m_0-1/m_15__to__m_16/MatMul:product:08L-15-m_0-1/m_15__to__m_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!L-15-m_0-1/m_15__to__m_16/BiasAdd?
!L-15-m_0-1/m_15__to__m_16/SoftmaxSoftmax%L-15-m_0-1/m_15__to__m_16/BiasAdd:z:0*
T0*'
_output_shapes
:?????????2#
!L-15-m_0-1/m_15__to__m_16/Softmax?
IdentityIdentity+L-15-m_0-1/m_15__to__m_16/Softmax:softmax:0/^L-15-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp.^L-15-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp1^L-15-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp0^L-15-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp1^L-15-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp0^L-15-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp1^L-15-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp0^L-15-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp1^L-15-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp0^L-15-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp1^L-15-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp0^L-15-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp1^L-15-m_0-1/m_15__to__m_16/BiasAdd/ReadVariableOp0^L-15-m_0-1/m_15__to__m_16/MatMul/ReadVariableOp/^L-15-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp.^L-15-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp/^L-15-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp.^L-15-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp/^L-15-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp.^L-15-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp/^L-15-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp.^L-15-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp/^L-15-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp.^L-15-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp/^L-15-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp.^L-15-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp/^L-15-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp.^L-15-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp/^L-15-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp.^L-15-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp0^L-15-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp/^L-15-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.L-15-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp.L-15-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp2^
-L-15-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp-L-15-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp2d
0L-15-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp0L-15-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp2b
/L-15-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp/L-15-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp2d
0L-15-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp0L-15-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp2b
/L-15-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp/L-15-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp2d
0L-15-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp0L-15-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp2b
/L-15-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp/L-15-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp2d
0L-15-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp0L-15-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp2b
/L-15-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp/L-15-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp2d
0L-15-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp0L-15-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp2b
/L-15-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp/L-15-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp2d
0L-15-m_0-1/m_15__to__m_16/BiasAdd/ReadVariableOp0L-15-m_0-1/m_15__to__m_16/BiasAdd/ReadVariableOp2b
/L-15-m_0-1/m_15__to__m_16/MatMul/ReadVariableOp/L-15-m_0-1/m_15__to__m_16/MatMul/ReadVariableOp2`
.L-15-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp.L-15-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp2^
-L-15-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp-L-15-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp2`
.L-15-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp.L-15-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp2^
-L-15-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp-L-15-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp2`
.L-15-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp.L-15-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp2^
-L-15-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp-L-15-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp2`
.L-15-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp.L-15-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp2^
-L-15-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp-L-15-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp2`
.L-15-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp.L-15-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp2^
-L-15-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp-L-15-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp2`
.L-15-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp.L-15-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp2^
-L-15-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp-L-15-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp2`
.L-15-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp.L-15-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp2^
-L-15-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp-L-15-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp2`
.L-15-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp.L-15-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp2^
-L-15-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp-L-15-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp2b
/L-15-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp/L-15-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp2`
.L-15-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp.L-15-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
__inference_loss_fn_3_13992246M
;m_3__to__m_4_kernel_regularizer_abs_readvariableop_resource:>>
identity??2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_3__to__m_4_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_3__to__m_4/kernel/Regularizer/Abs?
%m_3__to__m_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_3__to__m_4/kernel/Regularizer/Const?
#m_3__to__m_4/kernel/Regularizer/SumSum'm_3__to__m_4/kernel/Regularizer/Abs:y:0.m_3__to__m_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/Sum?
%m_3__to__m_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_3__to__m_4/kernel/Regularizer/mul/x?
#m_3__to__m_4/kernel/Regularizer/mulMul.m_3__to__m_4/kernel/Regularizer/mul/x:output:0,m_3__to__m_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/mul?
IdentityIdentity'm_3__to__m_4/kernel/Regularizer/mul:z:03^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp
?
?
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_13989671

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_6__to__m_7/kernel/Regularizer/Abs?
%m_6__to__m_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_6__to__m_7/kernel/Regularizer/Const?
#m_6__to__m_7/kernel/Regularizer/SumSum'm_6__to__m_7/kernel/Regularizer/Abs:y:0.m_6__to__m_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/Sum?
%m_6__to__m_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_6__to__m_7/kernel/Regularizer/mul/x?
#m_6__to__m_7/kernel/Regularizer/mulMul.m_6__to__m_7/kernel/Regularizer/mul/x:output:0,m_6__to__m_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
/__inference_m_1__to__m_2_layer_call_fn_13991754

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_139895562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_13989556

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_1__to__m_2/kernel/Regularizer/Abs?
%m_1__to__m_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_1__to__m_2/kernel/Regularizer/Const?
#m_1__to__m_2/kernel/Regularizer/SumSum'm_1__to__m_2/kernel/Regularizer/Abs:y:0.m_1__to__m_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/Sum?
%m_1__to__m_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_1__to__m_2/kernel/Regularizer/mul/x?
#m_1__to__m_2/kernel/Regularizer/mulMul.m_1__to__m_2/kernel/Regularizer/mul/x:output:0,m_1__to__m_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
/__inference_m_3__to__m_4_layer_call_fn_13991818

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_139896022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_13992224M
;m_1__to__m_2_kernel_regularizer_abs_readvariableop_resource:>>
identity??2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_1__to__m_2_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_1__to__m_2/kernel/Regularizer/Abs?
%m_1__to__m_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_1__to__m_2/kernel/Regularizer/Const?
#m_1__to__m_2/kernel/Regularizer/SumSum'm_1__to__m_2/kernel/Regularizer/Abs:y:0.m_1__to__m_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/Sum?
%m_1__to__m_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_1__to__m_2/kernel/Regularizer/mul/x?
#m_1__to__m_2/kernel/Regularizer/mulMul.m_1__to__m_2/kernel/Regularizer/mul/x:output:0,m_1__to__m_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/mul?
IdentityIdentity'm_1__to__m_2/kernel/Regularizer/mul:z:03^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp
?
?
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_13989602

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_3__to__m_4/kernel/Regularizer/Abs?
%m_3__to__m_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_3__to__m_4/kernel/Regularizer/Const?
#m_3__to__m_4/kernel/Regularizer/SumSum'm_3__to__m_4/kernel/Regularizer/Abs:y:0.m_3__to__m_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/Sum?
%m_3__to__m_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_3__to__m_4/kernel/Regularizer/mul/x?
#m_3__to__m_4/kernel/Regularizer/mulMul.m_3__to__m_4/kernel/Regularizer/mul/x:output:0,m_3__to__m_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_13992161

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_14__to__m_15/kernel/Regularizer/Abs?
'm_14__to__m_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_14__to__m_15/kernel/Regularizer/Const?
%m_14__to__m_15/kernel/Regularizer/SumSum)m_14__to__m_15/kernel/Regularizer/Abs:y:00m_14__to__m_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/Sum?
'm_14__to__m_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_14__to__m_15/kernel/Regularizer/mul/x?
%m_14__to__m_15/kernel/Regularizer/mulMul0m_14__to__m_15/kernel/Regularizer/mul/x:output:0.m_14__to__m_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
-__inference_L-15-m_0-1_layer_call_fn_13991690

inputs
unknown:>
	unknown_0:>
	unknown_1:>>
	unknown_2:>
	unknown_3:>>
	unknown_4:>
	unknown_5:>>
	unknown_6:>
	unknown_7:>>
	unknown_8:>
	unknown_9:>>

unknown_10:>

unknown_11:>>

unknown_12:>

unknown_13:>>

unknown_14:>

unknown_15:>>

unknown_16:>

unknown_17:>>

unknown_18:>

unknown_19:>>

unknown_20:>

unknown_21:>>

unknown_22:>

unknown_23:>>

unknown_24:>

unknown_25:>>

unknown_26:>

unknown_27:>>

unknown_28:>

unknown_29:>

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_139904592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_m_5__to__m_6_layer_call_fn_13991882

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_139896482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
0__inference_m_9__to__m_10_layer_call_fn_13992010

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_139897402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_13992257M
;m_4__to__m_5_kernel_regularizer_abs_readvariableop_resource:>>
identity??2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_4__to__m_5_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_4__to__m_5/kernel/Regularizer/Abs?
%m_4__to__m_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_4__to__m_5/kernel/Regularizer/Const?
#m_4__to__m_5/kernel/Regularizer/SumSum'm_4__to__m_5/kernel/Regularizer/Abs:y:0.m_4__to__m_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/Sum?
%m_4__to__m_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_4__to__m_5/kernel/Regularizer/mul/x?
#m_4__to__m_5/kernel/Regularizer/mulMul.m_4__to__m_5/kernel/Regularizer/mul/x:output:0,m_4__to__m_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/mul?
IdentityIdentity'm_4__to__m_5/kernel/Regularizer/mul:z:03^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp
?
?
-__inference_L-15-m_0-1_layer_call_fn_13990048
m_0__to__m_1_input
unknown:>
	unknown_0:>
	unknown_1:>>
	unknown_2:>
	unknown_3:>>
	unknown_4:>
	unknown_5:>>
	unknown_6:>
	unknown_7:>>
	unknown_8:>
	unknown_9:>>

unknown_10:>

unknown_11:>>

unknown_12:>

unknown_13:>>

unknown_14:>

unknown_15:>>

unknown_16:>

unknown_17:>>

unknown_18:>

unknown_19:>>

unknown_20:>

unknown_21:>>

unknown_22:>

unknown_23:>>

unknown_24:>

unknown_25:>>

unknown_26:>

unknown_27:>>

unknown_28:>

unknown_29:>

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallm_0__to__m_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_139899812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_13989579

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_2__to__m_3/kernel/Regularizer/Abs?
%m_2__to__m_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_2__to__m_3/kernel/Regularizer/Const?
#m_2__to__m_3/kernel/Regularizer/SumSum'm_2__to__m_3/kernel/Regularizer/Abs:y:0.m_2__to__m_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/Sum?
%m_2__to__m_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_2__to__m_3/kernel/Regularizer/mul/x?
#m_2__to__m_3/kernel/Regularizer/mulMul.m_2__to__m_3/kernel/Regularizer/mul/x:output:0,m_2__to__m_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_13989809

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_12__to__m_13/kernel/Regularizer/Abs?
'm_12__to__m_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_12__to__m_13/kernel/Regularizer/Const?
%m_12__to__m_13/kernel/Regularizer/SumSum)m_12__to__m_13/kernel/Regularizer/Abs:y:00m_12__to__m_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/Sum?
'm_12__to__m_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_12__to__m_13/kernel/Regularizer/mul/x?
%m_12__to__m_13/kernel/Regularizer/mulMul0m_12__to__m_13/kernel/Regularizer/mul/x:output:0.m_12__to__m_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
??
?
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13989981

inputs'
m_0__to__m_1_13989534:>#
m_0__to__m_1_13989536:>'
m_1__to__m_2_13989557:>>#
m_1__to__m_2_13989559:>'
m_2__to__m_3_13989580:>>#
m_2__to__m_3_13989582:>'
m_3__to__m_4_13989603:>>#
m_3__to__m_4_13989605:>'
m_4__to__m_5_13989626:>>#
m_4__to__m_5_13989628:>'
m_5__to__m_6_13989649:>>#
m_5__to__m_6_13989651:>'
m_6__to__m_7_13989672:>>#
m_6__to__m_7_13989674:>'
m_7__to__m_8_13989695:>>#
m_7__to__m_8_13989697:>'
m_8__to__m_9_13989718:>>#
m_8__to__m_9_13989720:>(
m_9__to__m_10_13989741:>>$
m_9__to__m_10_13989743:>)
m_10__to__m_11_13989764:>>%
m_10__to__m_11_13989766:>)
m_11__to__m_12_13989787:>>%
m_11__to__m_12_13989789:>)
m_12__to__m_13_13989810:>>%
m_12__to__m_13_13989812:>)
m_13__to__m_14_13989833:>>%
m_13__to__m_14_13989835:>)
m_14__to__m_15_13989856:>>%
m_14__to__m_15_13989858:>)
m_15__to__m_16_13989879:>%
m_15__to__m_16_13989881:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?&m_15__to__m_16/StatefulPartitionedCall?4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallinputsm_0__to__m_1_13989534m_0__to__m_1_13989536*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_139895332&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_13989557m_1__to__m_2_13989559*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_139895562&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_13989580m_2__to__m_3_13989582*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_139895792&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_13989603m_3__to__m_4_13989605*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_139896022&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_13989626m_4__to__m_5_13989628*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_139896252&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_13989649m_5__to__m_6_13989651*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_139896482&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_13989672m_6__to__m_7_13989674*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_139896712&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_13989695m_7__to__m_8_13989697*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_139896942&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_13989718m_8__to__m_9_13989720*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_139897172&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_13989741m_9__to__m_10_13989743*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_139897402'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_13989764m_10__to__m_11_13989766*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_139897632(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_13989787m_11__to__m_12_13989789*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_139897862(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_13989810m_12__to__m_13_13989812*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_139898092(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_13989833m_13__to__m_14_13989835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_139898322(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_13989856m_14__to__m_15_13989858*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_139898552(
&m_14__to__m_15/StatefulPartitionedCall?
&m_15__to__m_16/StatefulPartitionedCallStatefulPartitionedCall/m_14__to__m_15/StatefulPartitionedCall:output:0m_15__to__m_16_13989879m_15__to__m_16_13989881*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_15__to__m_16_layer_call_and_return_conditional_losses_139898782(
&m_15__to__m_16/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_13989534*
_output_shapes

:>*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2%
#m_0__to__m_1/kernel/Regularizer/Abs?
%m_0__to__m_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_0__to__m_1/kernel/Regularizer/Const?
#m_0__to__m_1/kernel/Regularizer/SumSum'm_0__to__m_1/kernel/Regularizer/Abs:y:0.m_0__to__m_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/Sum?
%m_0__to__m_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_0__to__m_1/kernel/Regularizer/mul/x?
#m_0__to__m_1/kernel/Regularizer/mulMul.m_0__to__m_1/kernel/Regularizer/mul/x:output:0,m_0__to__m_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/mul?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_13989557*
_output_shapes

:>>*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_1__to__m_2/kernel/Regularizer/Abs?
%m_1__to__m_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_1__to__m_2/kernel/Regularizer/Const?
#m_1__to__m_2/kernel/Regularizer/SumSum'm_1__to__m_2/kernel/Regularizer/Abs:y:0.m_1__to__m_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/Sum?
%m_1__to__m_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_1__to__m_2/kernel/Regularizer/mul/x?
#m_1__to__m_2/kernel/Regularizer/mulMul.m_1__to__m_2/kernel/Regularizer/mul/x:output:0,m_1__to__m_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/mul?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_13989580*
_output_shapes

:>>*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_2__to__m_3/kernel/Regularizer/Abs?
%m_2__to__m_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_2__to__m_3/kernel/Regularizer/Const?
#m_2__to__m_3/kernel/Regularizer/SumSum'm_2__to__m_3/kernel/Regularizer/Abs:y:0.m_2__to__m_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/Sum?
%m_2__to__m_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_2__to__m_3/kernel/Regularizer/mul/x?
#m_2__to__m_3/kernel/Regularizer/mulMul.m_2__to__m_3/kernel/Regularizer/mul/x:output:0,m_2__to__m_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/mul?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_13989603*
_output_shapes

:>>*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_3__to__m_4/kernel/Regularizer/Abs?
%m_3__to__m_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_3__to__m_4/kernel/Regularizer/Const?
#m_3__to__m_4/kernel/Regularizer/SumSum'm_3__to__m_4/kernel/Regularizer/Abs:y:0.m_3__to__m_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/Sum?
%m_3__to__m_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_3__to__m_4/kernel/Regularizer/mul/x?
#m_3__to__m_4/kernel/Regularizer/mulMul.m_3__to__m_4/kernel/Regularizer/mul/x:output:0,m_3__to__m_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/mul?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_13989626*
_output_shapes

:>>*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_4__to__m_5/kernel/Regularizer/Abs?
%m_4__to__m_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_4__to__m_5/kernel/Regularizer/Const?
#m_4__to__m_5/kernel/Regularizer/SumSum'm_4__to__m_5/kernel/Regularizer/Abs:y:0.m_4__to__m_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/Sum?
%m_4__to__m_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_4__to__m_5/kernel/Regularizer/mul/x?
#m_4__to__m_5/kernel/Regularizer/mulMul.m_4__to__m_5/kernel/Regularizer/mul/x:output:0,m_4__to__m_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/mul?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_13989649*
_output_shapes

:>>*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_5__to__m_6/kernel/Regularizer/Abs?
%m_5__to__m_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_5__to__m_6/kernel/Regularizer/Const?
#m_5__to__m_6/kernel/Regularizer/SumSum'm_5__to__m_6/kernel/Regularizer/Abs:y:0.m_5__to__m_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/Sum?
%m_5__to__m_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_5__to__m_6/kernel/Regularizer/mul/x?
#m_5__to__m_6/kernel/Regularizer/mulMul.m_5__to__m_6/kernel/Regularizer/mul/x:output:0,m_5__to__m_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/mul?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_13989672*
_output_shapes

:>>*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_6__to__m_7/kernel/Regularizer/Abs?
%m_6__to__m_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_6__to__m_7/kernel/Regularizer/Const?
#m_6__to__m_7/kernel/Regularizer/SumSum'm_6__to__m_7/kernel/Regularizer/Abs:y:0.m_6__to__m_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/Sum?
%m_6__to__m_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_6__to__m_7/kernel/Regularizer/mul/x?
#m_6__to__m_7/kernel/Regularizer/mulMul.m_6__to__m_7/kernel/Regularizer/mul/x:output:0,m_6__to__m_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/mul?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_13989695*
_output_shapes

:>>*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_7__to__m_8/kernel/Regularizer/Abs?
%m_7__to__m_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_7__to__m_8/kernel/Regularizer/Const?
#m_7__to__m_8/kernel/Regularizer/SumSum'm_7__to__m_8/kernel/Regularizer/Abs:y:0.m_7__to__m_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/Sum?
%m_7__to__m_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_7__to__m_8/kernel/Regularizer/mul/x?
#m_7__to__m_8/kernel/Regularizer/mulMul.m_7__to__m_8/kernel/Regularizer/mul/x:output:0,m_7__to__m_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/mul?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_13989718*
_output_shapes

:>>*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_8__to__m_9/kernel/Regularizer/Abs?
%m_8__to__m_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_8__to__m_9/kernel/Regularizer/Const?
#m_8__to__m_9/kernel/Regularizer/SumSum'm_8__to__m_9/kernel/Regularizer/Abs:y:0.m_8__to__m_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/Sum?
%m_8__to__m_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_8__to__m_9/kernel/Regularizer/mul/x?
#m_8__to__m_9/kernel/Regularizer/mulMul.m_8__to__m_9/kernel/Regularizer/mul/x:output:0,m_8__to__m_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/mul?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_13989741*
_output_shapes

:>>*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2&
$m_9__to__m_10/kernel/Regularizer/Abs?
&m_9__to__m_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&m_9__to__m_10/kernel/Regularizer/Const?
$m_9__to__m_10/kernel/Regularizer/SumSum(m_9__to__m_10/kernel/Regularizer/Abs:y:0/m_9__to__m_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/Sum?
&m_9__to__m_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92(
&m_9__to__m_10/kernel/Regularizer/mul/x?
$m_9__to__m_10/kernel/Regularizer/mulMul/m_9__to__m_10/kernel/Regularizer/mul/x:output:0-m_9__to__m_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/mul?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_13989764*
_output_shapes

:>>*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_10__to__m_11/kernel/Regularizer/Abs?
'm_10__to__m_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_10__to__m_11/kernel/Regularizer/Const?
%m_10__to__m_11/kernel/Regularizer/SumSum)m_10__to__m_11/kernel/Regularizer/Abs:y:00m_10__to__m_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/Sum?
'm_10__to__m_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_10__to__m_11/kernel/Regularizer/mul/x?
%m_10__to__m_11/kernel/Regularizer/mulMul0m_10__to__m_11/kernel/Regularizer/mul/x:output:0.m_10__to__m_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/mul?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_13989787*
_output_shapes

:>>*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_11__to__m_12/kernel/Regularizer/Abs?
'm_11__to__m_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_11__to__m_12/kernel/Regularizer/Const?
%m_11__to__m_12/kernel/Regularizer/SumSum)m_11__to__m_12/kernel/Regularizer/Abs:y:00m_11__to__m_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/Sum?
'm_11__to__m_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_11__to__m_12/kernel/Regularizer/mul/x?
%m_11__to__m_12/kernel/Regularizer/mulMul0m_11__to__m_12/kernel/Regularizer/mul/x:output:0.m_11__to__m_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/mul?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_13989810*
_output_shapes

:>>*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_12__to__m_13/kernel/Regularizer/Abs?
'm_12__to__m_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_12__to__m_13/kernel/Regularizer/Const?
%m_12__to__m_13/kernel/Regularizer/SumSum)m_12__to__m_13/kernel/Regularizer/Abs:y:00m_12__to__m_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/Sum?
'm_12__to__m_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_12__to__m_13/kernel/Regularizer/mul/x?
%m_12__to__m_13/kernel/Regularizer/mulMul0m_12__to__m_13/kernel/Regularizer/mul/x:output:0.m_12__to__m_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/mul?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_13989833*
_output_shapes

:>>*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_13__to__m_14/kernel/Regularizer/Abs?
'm_13__to__m_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_13__to__m_14/kernel/Regularizer/Const?
%m_13__to__m_14/kernel/Regularizer/SumSum)m_13__to__m_14/kernel/Regularizer/Abs:y:00m_13__to__m_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/Sum?
'm_13__to__m_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_13__to__m_14/kernel/Regularizer/mul/x?
%m_13__to__m_14/kernel/Regularizer/mulMul0m_13__to__m_14/kernel/Regularizer/mul/x:output:0.m_13__to__m_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/mul?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_13989856*
_output_shapes

:>>*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_14__to__m_15/kernel/Regularizer/Abs?
'm_14__to__m_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_14__to__m_15/kernel/Regularizer/Const?
%m_14__to__m_15/kernel/Regularizer/SumSum)m_14__to__m_15/kernel/Regularizer/Abs:y:00m_14__to__m_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/Sum?
'm_14__to__m_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_14__to__m_15/kernel/Regularizer/mul/x?
%m_14__to__m_15/kernel/Regularizer/mulMul0m_14__to__m_15/kernel/Regularizer/mul/x:output:0.m_14__to__m_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/mul?
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_15__to__m_16_13989879*
_output_shapes

:>*
dtype026
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
%m_15__to__m_16/kernel/Regularizer/AbsAbs<m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2'
%m_15__to__m_16/kernel/Regularizer/Abs?
'm_15__to__m_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_15__to__m_16/kernel/Regularizer/Const?
%m_15__to__m_16/kernel/Regularizer/SumSum)m_15__to__m_16/kernel/Regularizer/Abs:y:00m_15__to__m_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/Sum?
'm_15__to__m_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_15__to__m_16/kernel/Regularizer/mul/x?
%m_15__to__m_16/kernel/Regularizer/mulMul0m_15__to__m_16/kernel/Regularizer/mul/x:output:0.m_15__to__m_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/mul?
IdentityIdentity/m_15__to__m_16/StatefulPartitionedCall:output:0%^m_0__to__m_1/StatefulPartitionedCall3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp'^m_10__to__m_11/StatefulPartitionedCall5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp'^m_11__to__m_12/StatefulPartitionedCall5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp'^m_12__to__m_13/StatefulPartitionedCall5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp'^m_13__to__m_14/StatefulPartitionedCall5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp'^m_14__to__m_15/StatefulPartitionedCall5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp'^m_15__to__m_16/StatefulPartitionedCall5^m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp%^m_1__to__m_2/StatefulPartitionedCall3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp%^m_2__to__m_3/StatefulPartitionedCall3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp%^m_3__to__m_4/StatefulPartitionedCall3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp%^m_4__to__m_5/StatefulPartitionedCall3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp%^m_5__to__m_6/StatefulPartitionedCall3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp%^m_6__to__m_7/StatefulPartitionedCall3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp%^m_7__to__m_8/StatefulPartitionedCall3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp%^m_8__to__m_9/StatefulPartitionedCall3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp&^m_9__to__m_10/StatefulPartitionedCall4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$m_0__to__m_1/StatefulPartitionedCall$m_0__to__m_1/StatefulPartitionedCall2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2P
&m_10__to__m_11/StatefulPartitionedCall&m_10__to__m_11/StatefulPartitionedCall2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp2P
&m_11__to__m_12/StatefulPartitionedCall&m_11__to__m_12/StatefulPartitionedCall2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp2P
&m_12__to__m_13/StatefulPartitionedCall&m_12__to__m_13/StatefulPartitionedCall2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp2P
&m_13__to__m_14/StatefulPartitionedCall&m_13__to__m_14/StatefulPartitionedCall2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp2P
&m_14__to__m_15/StatefulPartitionedCall&m_14__to__m_15/StatefulPartitionedCall2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2P
&m_15__to__m_16/StatefulPartitionedCall&m_15__to__m_16/StatefulPartitionedCall2l
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp2L
$m_1__to__m_2/StatefulPartitionedCall$m_1__to__m_2/StatefulPartitionedCall2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2L
$m_2__to__m_3/StatefulPartitionedCall$m_2__to__m_3/StatefulPartitionedCall2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2L
$m_3__to__m_4/StatefulPartitionedCall$m_3__to__m_4/StatefulPartitionedCall2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2L
$m_4__to__m_5/StatefulPartitionedCall$m_4__to__m_5/StatefulPartitionedCall2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2L
$m_5__to__m_6/StatefulPartitionedCall$m_5__to__m_6/StatefulPartitionedCall2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2L
$m_6__to__m_7/StatefulPartitionedCall$m_6__to__m_7/StatefulPartitionedCall2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2L
$m_7__to__m_8/StatefulPartitionedCall$m_7__to__m_8/StatefulPartitionedCall2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2L
$m_8__to__m_9/StatefulPartitionedCall$m_8__to__m_9/StatefulPartitionedCall2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2N
%m_9__to__m_10/StatefulPartitionedCall%m_9__to__m_10/StatefulPartitionedCall2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_13_13992356O
=m_13__to__m_14_kernel_regularizer_abs_readvariableop_resource:>>
identity??4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_13__to__m_14_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_13__to__m_14/kernel/Regularizer/Abs?
'm_13__to__m_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_13__to__m_14/kernel/Regularizer/Const?
%m_13__to__m_14/kernel/Regularizer/SumSum)m_13__to__m_14/kernel/Regularizer/Abs:y:00m_13__to__m_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/Sum?
'm_13__to__m_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_13__to__m_14/kernel/Regularizer/mul/x?
%m_13__to__m_14/kernel/Regularizer/mulMul0m_13__to__m_14/kernel/Regularizer/mul/x:output:0.m_13__to__m_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/mul?
IdentityIdentity)m_13__to__m_14/kernel/Regularizer/mul:z:05^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp
?
?
-__inference_L-15-m_0-1_layer_call_fn_13990595
m_0__to__m_1_input
unknown:>
	unknown_0:>
	unknown_1:>>
	unknown_2:>
	unknown_3:>>
	unknown_4:>
	unknown_5:>>
	unknown_6:>
	unknown_7:>>
	unknown_8:>
	unknown_9:>>

unknown_10:>

unknown_11:>>

unknown_12:>

unknown_13:>>

unknown_14:>

unknown_15:>>

unknown_16:>

unknown_17:>>

unknown_18:>

unknown_19:>>

unknown_20:>

unknown_21:>>

unknown_22:>

unknown_23:>>

unknown_24:>

unknown_25:>>

unknown_26:>

unknown_27:>>

unknown_28:>

unknown_29:>

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallm_0__to__m_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_139904592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
/__inference_m_6__to__m_7_layer_call_fn_13991914

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_139896712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
1__inference_m_10__to__m_11_layer_call_fn_13992042

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_139897632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_13991969

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_8__to__m_9/kernel/Regularizer/Abs?
%m_8__to__m_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_8__to__m_9/kernel/Regularizer/Const?
#m_8__to__m_9/kernel/Regularizer/SumSum'm_8__to__m_9/kernel/Regularizer/Abs:y:0.m_8__to__m_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/Sum?
%m_8__to__m_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_8__to__m_9/kernel/Regularizer/mul/x?
#m_8__to__m_9/kernel/Regularizer/mulMul.m_8__to__m_9/kernel/Regularizer/mul/x:output:0,m_8__to__m_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
??
?
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13990955
m_0__to__m_1_input'
m_0__to__m_1_13990778:>#
m_0__to__m_1_13990780:>'
m_1__to__m_2_13990783:>>#
m_1__to__m_2_13990785:>'
m_2__to__m_3_13990788:>>#
m_2__to__m_3_13990790:>'
m_3__to__m_4_13990793:>>#
m_3__to__m_4_13990795:>'
m_4__to__m_5_13990798:>>#
m_4__to__m_5_13990800:>'
m_5__to__m_6_13990803:>>#
m_5__to__m_6_13990805:>'
m_6__to__m_7_13990808:>>#
m_6__to__m_7_13990810:>'
m_7__to__m_8_13990813:>>#
m_7__to__m_8_13990815:>'
m_8__to__m_9_13990818:>>#
m_8__to__m_9_13990820:>(
m_9__to__m_10_13990823:>>$
m_9__to__m_10_13990825:>)
m_10__to__m_11_13990828:>>%
m_10__to__m_11_13990830:>)
m_11__to__m_12_13990833:>>%
m_11__to__m_12_13990835:>)
m_12__to__m_13_13990838:>>%
m_12__to__m_13_13990840:>)
m_13__to__m_14_13990843:>>%
m_13__to__m_14_13990845:>)
m_14__to__m_15_13990848:>>%
m_14__to__m_15_13990850:>)
m_15__to__m_16_13990853:>%
m_15__to__m_16_13990855:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?&m_15__to__m_16/StatefulPartitionedCall?4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallm_0__to__m_1_inputm_0__to__m_1_13990778m_0__to__m_1_13990780*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_139895332&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_13990783m_1__to__m_2_13990785*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_139895562&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_13990788m_2__to__m_3_13990790*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_139895792&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_13990793m_3__to__m_4_13990795*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_139896022&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_13990798m_4__to__m_5_13990800*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_139896252&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_13990803m_5__to__m_6_13990805*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_139896482&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_13990808m_6__to__m_7_13990810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_139896712&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_13990813m_7__to__m_8_13990815*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_139896942&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_13990818m_8__to__m_9_13990820*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_139897172&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_13990823m_9__to__m_10_13990825*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_139897402'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_13990828m_10__to__m_11_13990830*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_139897632(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_13990833m_11__to__m_12_13990835*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_139897862(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_13990838m_12__to__m_13_13990840*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_139898092(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_13990843m_13__to__m_14_13990845*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_139898322(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_13990848m_14__to__m_15_13990850*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_139898552(
&m_14__to__m_15/StatefulPartitionedCall?
&m_15__to__m_16/StatefulPartitionedCallStatefulPartitionedCall/m_14__to__m_15/StatefulPartitionedCall:output:0m_15__to__m_16_13990853m_15__to__m_16_13990855*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_15__to__m_16_layer_call_and_return_conditional_losses_139898782(
&m_15__to__m_16/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_13990778*
_output_shapes

:>*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2%
#m_0__to__m_1/kernel/Regularizer/Abs?
%m_0__to__m_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_0__to__m_1/kernel/Regularizer/Const?
#m_0__to__m_1/kernel/Regularizer/SumSum'm_0__to__m_1/kernel/Regularizer/Abs:y:0.m_0__to__m_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/Sum?
%m_0__to__m_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_0__to__m_1/kernel/Regularizer/mul/x?
#m_0__to__m_1/kernel/Regularizer/mulMul.m_0__to__m_1/kernel/Regularizer/mul/x:output:0,m_0__to__m_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/mul?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_13990783*
_output_shapes

:>>*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_1__to__m_2/kernel/Regularizer/Abs?
%m_1__to__m_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_1__to__m_2/kernel/Regularizer/Const?
#m_1__to__m_2/kernel/Regularizer/SumSum'm_1__to__m_2/kernel/Regularizer/Abs:y:0.m_1__to__m_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/Sum?
%m_1__to__m_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_1__to__m_2/kernel/Regularizer/mul/x?
#m_1__to__m_2/kernel/Regularizer/mulMul.m_1__to__m_2/kernel/Regularizer/mul/x:output:0,m_1__to__m_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/mul?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_13990788*
_output_shapes

:>>*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_2__to__m_3/kernel/Regularizer/Abs?
%m_2__to__m_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_2__to__m_3/kernel/Regularizer/Const?
#m_2__to__m_3/kernel/Regularizer/SumSum'm_2__to__m_3/kernel/Regularizer/Abs:y:0.m_2__to__m_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/Sum?
%m_2__to__m_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_2__to__m_3/kernel/Regularizer/mul/x?
#m_2__to__m_3/kernel/Regularizer/mulMul.m_2__to__m_3/kernel/Regularizer/mul/x:output:0,m_2__to__m_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/mul?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_13990793*
_output_shapes

:>>*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_3__to__m_4/kernel/Regularizer/Abs?
%m_3__to__m_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_3__to__m_4/kernel/Regularizer/Const?
#m_3__to__m_4/kernel/Regularizer/SumSum'm_3__to__m_4/kernel/Regularizer/Abs:y:0.m_3__to__m_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/Sum?
%m_3__to__m_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_3__to__m_4/kernel/Regularizer/mul/x?
#m_3__to__m_4/kernel/Regularizer/mulMul.m_3__to__m_4/kernel/Regularizer/mul/x:output:0,m_3__to__m_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/mul?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_13990798*
_output_shapes

:>>*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_4__to__m_5/kernel/Regularizer/Abs?
%m_4__to__m_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_4__to__m_5/kernel/Regularizer/Const?
#m_4__to__m_5/kernel/Regularizer/SumSum'm_4__to__m_5/kernel/Regularizer/Abs:y:0.m_4__to__m_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/Sum?
%m_4__to__m_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_4__to__m_5/kernel/Regularizer/mul/x?
#m_4__to__m_5/kernel/Regularizer/mulMul.m_4__to__m_5/kernel/Regularizer/mul/x:output:0,m_4__to__m_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/mul?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_13990803*
_output_shapes

:>>*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_5__to__m_6/kernel/Regularizer/Abs?
%m_5__to__m_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_5__to__m_6/kernel/Regularizer/Const?
#m_5__to__m_6/kernel/Regularizer/SumSum'm_5__to__m_6/kernel/Regularizer/Abs:y:0.m_5__to__m_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/Sum?
%m_5__to__m_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_5__to__m_6/kernel/Regularizer/mul/x?
#m_5__to__m_6/kernel/Regularizer/mulMul.m_5__to__m_6/kernel/Regularizer/mul/x:output:0,m_5__to__m_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/mul?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_13990808*
_output_shapes

:>>*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_6__to__m_7/kernel/Regularizer/Abs?
%m_6__to__m_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_6__to__m_7/kernel/Regularizer/Const?
#m_6__to__m_7/kernel/Regularizer/SumSum'm_6__to__m_7/kernel/Regularizer/Abs:y:0.m_6__to__m_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/Sum?
%m_6__to__m_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_6__to__m_7/kernel/Regularizer/mul/x?
#m_6__to__m_7/kernel/Regularizer/mulMul.m_6__to__m_7/kernel/Regularizer/mul/x:output:0,m_6__to__m_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/mul?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_13990813*
_output_shapes

:>>*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_7__to__m_8/kernel/Regularizer/Abs?
%m_7__to__m_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_7__to__m_8/kernel/Regularizer/Const?
#m_7__to__m_8/kernel/Regularizer/SumSum'm_7__to__m_8/kernel/Regularizer/Abs:y:0.m_7__to__m_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/Sum?
%m_7__to__m_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_7__to__m_8/kernel/Regularizer/mul/x?
#m_7__to__m_8/kernel/Regularizer/mulMul.m_7__to__m_8/kernel/Regularizer/mul/x:output:0,m_7__to__m_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/mul?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_13990818*
_output_shapes

:>>*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_8__to__m_9/kernel/Regularizer/Abs?
%m_8__to__m_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_8__to__m_9/kernel/Regularizer/Const?
#m_8__to__m_9/kernel/Regularizer/SumSum'm_8__to__m_9/kernel/Regularizer/Abs:y:0.m_8__to__m_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/Sum?
%m_8__to__m_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_8__to__m_9/kernel/Regularizer/mul/x?
#m_8__to__m_9/kernel/Regularizer/mulMul.m_8__to__m_9/kernel/Regularizer/mul/x:output:0,m_8__to__m_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/mul?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_13990823*
_output_shapes

:>>*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2&
$m_9__to__m_10/kernel/Regularizer/Abs?
&m_9__to__m_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&m_9__to__m_10/kernel/Regularizer/Const?
$m_9__to__m_10/kernel/Regularizer/SumSum(m_9__to__m_10/kernel/Regularizer/Abs:y:0/m_9__to__m_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/Sum?
&m_9__to__m_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92(
&m_9__to__m_10/kernel/Regularizer/mul/x?
$m_9__to__m_10/kernel/Regularizer/mulMul/m_9__to__m_10/kernel/Regularizer/mul/x:output:0-m_9__to__m_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/mul?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_13990828*
_output_shapes

:>>*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_10__to__m_11/kernel/Regularizer/Abs?
'm_10__to__m_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_10__to__m_11/kernel/Regularizer/Const?
%m_10__to__m_11/kernel/Regularizer/SumSum)m_10__to__m_11/kernel/Regularizer/Abs:y:00m_10__to__m_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/Sum?
'm_10__to__m_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_10__to__m_11/kernel/Regularizer/mul/x?
%m_10__to__m_11/kernel/Regularizer/mulMul0m_10__to__m_11/kernel/Regularizer/mul/x:output:0.m_10__to__m_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/mul?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_13990833*
_output_shapes

:>>*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_11__to__m_12/kernel/Regularizer/Abs?
'm_11__to__m_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_11__to__m_12/kernel/Regularizer/Const?
%m_11__to__m_12/kernel/Regularizer/SumSum)m_11__to__m_12/kernel/Regularizer/Abs:y:00m_11__to__m_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/Sum?
'm_11__to__m_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_11__to__m_12/kernel/Regularizer/mul/x?
%m_11__to__m_12/kernel/Regularizer/mulMul0m_11__to__m_12/kernel/Regularizer/mul/x:output:0.m_11__to__m_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/mul?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_13990838*
_output_shapes

:>>*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_12__to__m_13/kernel/Regularizer/Abs?
'm_12__to__m_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_12__to__m_13/kernel/Regularizer/Const?
%m_12__to__m_13/kernel/Regularizer/SumSum)m_12__to__m_13/kernel/Regularizer/Abs:y:00m_12__to__m_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/Sum?
'm_12__to__m_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_12__to__m_13/kernel/Regularizer/mul/x?
%m_12__to__m_13/kernel/Regularizer/mulMul0m_12__to__m_13/kernel/Regularizer/mul/x:output:0.m_12__to__m_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/mul?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_13990843*
_output_shapes

:>>*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_13__to__m_14/kernel/Regularizer/Abs?
'm_13__to__m_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_13__to__m_14/kernel/Regularizer/Const?
%m_13__to__m_14/kernel/Regularizer/SumSum)m_13__to__m_14/kernel/Regularizer/Abs:y:00m_13__to__m_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/Sum?
'm_13__to__m_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_13__to__m_14/kernel/Regularizer/mul/x?
%m_13__to__m_14/kernel/Regularizer/mulMul0m_13__to__m_14/kernel/Regularizer/mul/x:output:0.m_13__to__m_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/mul?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_13990848*
_output_shapes

:>>*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_14__to__m_15/kernel/Regularizer/Abs?
'm_14__to__m_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_14__to__m_15/kernel/Regularizer/Const?
%m_14__to__m_15/kernel/Regularizer/SumSum)m_14__to__m_15/kernel/Regularizer/Abs:y:00m_14__to__m_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/Sum?
'm_14__to__m_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_14__to__m_15/kernel/Regularizer/mul/x?
%m_14__to__m_15/kernel/Regularizer/mulMul0m_14__to__m_15/kernel/Regularizer/mul/x:output:0.m_14__to__m_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/mul?
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_15__to__m_16_13990853*
_output_shapes

:>*
dtype026
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
%m_15__to__m_16/kernel/Regularizer/AbsAbs<m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2'
%m_15__to__m_16/kernel/Regularizer/Abs?
'm_15__to__m_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_15__to__m_16/kernel/Regularizer/Const?
%m_15__to__m_16/kernel/Regularizer/SumSum)m_15__to__m_16/kernel/Regularizer/Abs:y:00m_15__to__m_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/Sum?
'm_15__to__m_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_15__to__m_16/kernel/Regularizer/mul/x?
%m_15__to__m_16/kernel/Regularizer/mulMul0m_15__to__m_16/kernel/Regularizer/mul/x:output:0.m_15__to__m_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/mul?
IdentityIdentity/m_15__to__m_16/StatefulPartitionedCall:output:0%^m_0__to__m_1/StatefulPartitionedCall3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp'^m_10__to__m_11/StatefulPartitionedCall5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp'^m_11__to__m_12/StatefulPartitionedCall5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp'^m_12__to__m_13/StatefulPartitionedCall5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp'^m_13__to__m_14/StatefulPartitionedCall5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp'^m_14__to__m_15/StatefulPartitionedCall5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp'^m_15__to__m_16/StatefulPartitionedCall5^m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp%^m_1__to__m_2/StatefulPartitionedCall3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp%^m_2__to__m_3/StatefulPartitionedCall3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp%^m_3__to__m_4/StatefulPartitionedCall3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp%^m_4__to__m_5/StatefulPartitionedCall3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp%^m_5__to__m_6/StatefulPartitionedCall3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp%^m_6__to__m_7/StatefulPartitionedCall3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp%^m_7__to__m_8/StatefulPartitionedCall3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp%^m_8__to__m_9/StatefulPartitionedCall3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp&^m_9__to__m_10/StatefulPartitionedCall4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
$m_0__to__m_1/StatefulPartitionedCall$m_0__to__m_1/StatefulPartitionedCall2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2P
&m_10__to__m_11/StatefulPartitionedCall&m_10__to__m_11/StatefulPartitionedCall2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp2P
&m_11__to__m_12/StatefulPartitionedCall&m_11__to__m_12/StatefulPartitionedCall2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp2P
&m_12__to__m_13/StatefulPartitionedCall&m_12__to__m_13/StatefulPartitionedCall2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp2P
&m_13__to__m_14/StatefulPartitionedCall&m_13__to__m_14/StatefulPartitionedCall2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp2P
&m_14__to__m_15/StatefulPartitionedCall&m_14__to__m_15/StatefulPartitionedCall2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2P
&m_15__to__m_16/StatefulPartitionedCall&m_15__to__m_16/StatefulPartitionedCall2l
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp2L
$m_1__to__m_2/StatefulPartitionedCall$m_1__to__m_2/StatefulPartitionedCall2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2L
$m_2__to__m_3/StatefulPartitionedCall$m_2__to__m_3/StatefulPartitionedCall2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2L
$m_3__to__m_4/StatefulPartitionedCall$m_3__to__m_4/StatefulPartitionedCall2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2L
$m_4__to__m_5/StatefulPartitionedCall$m_4__to__m_5/StatefulPartitionedCall2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2L
$m_5__to__m_6/StatefulPartitionedCall$m_5__to__m_6/StatefulPartitionedCall2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2L
$m_6__to__m_7/StatefulPartitionedCall$m_6__to__m_7/StatefulPartitionedCall2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2L
$m_7__to__m_8/StatefulPartitionedCall$m_7__to__m_8/StatefulPartitionedCall2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2L
$m_8__to__m_9/StatefulPartitionedCall$m_8__to__m_9/StatefulPartitionedCall2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2N
%m_9__to__m_10/StatefulPartitionedCall%m_9__to__m_10/StatefulPartitionedCall2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
__inference_loss_fn_7_13992290M
;m_7__to__m_8_kernel_regularizer_abs_readvariableop_resource:>>
identity??2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_7__to__m_8_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_7__to__m_8/kernel/Regularizer/Abs?
%m_7__to__m_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_7__to__m_8/kernel/Regularizer/Const?
#m_7__to__m_8/kernel/Regularizer/SumSum'm_7__to__m_8/kernel/Regularizer/Abs:y:0.m_7__to__m_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/Sum?
%m_7__to__m_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_7__to__m_8/kernel/Regularizer/mul/x?
#m_7__to__m_8/kernel/Regularizer/mulMul.m_7__to__m_8/kernel/Regularizer/mul/x:output:0,m_7__to__m_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/mul?
IdentityIdentity'm_7__to__m_8/kernel/Regularizer/mul:z:03^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp
?
?
__inference_loss_fn_8_13992301M
;m_8__to__m_9_kernel_regularizer_abs_readvariableop_resource:>>
identity??2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_8__to__m_9_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_8__to__m_9/kernel/Regularizer/Abs?
%m_8__to__m_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_8__to__m_9/kernel/Regularizer/Const?
#m_8__to__m_9/kernel/Regularizer/SumSum'm_8__to__m_9/kernel/Regularizer/Abs:y:0.m_8__to__m_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/Sum?
%m_8__to__m_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_8__to__m_9/kernel/Regularizer/mul/x?
#m_8__to__m_9/kernel/Regularizer/mulMul.m_8__to__m_9/kernel/Regularizer/mul/x:output:0,m_8__to__m_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/mul?
IdentityIdentity'm_8__to__m_9/kernel/Regularizer/mul:z:03^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp
?
?
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_13992001

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2&
$m_9__to__m_10/kernel/Regularizer/Abs?
&m_9__to__m_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&m_9__to__m_10/kernel/Regularizer/Const?
$m_9__to__m_10/kernel/Regularizer/SumSum(m_9__to__m_10/kernel/Regularizer/Abs:y:0/m_9__to__m_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/Sum?
&m_9__to__m_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92(
&m_9__to__m_10/kernel/Regularizer/mul/x?
$m_9__to__m_10/kernel/Regularizer/mulMul/m_9__to__m_10/kernel/Regularizer/mul/x:output:0-m_9__to__m_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
-__inference_L-15-m_0-1_layer_call_fn_13991621

inputs
unknown:>
	unknown_0:>
	unknown_1:>>
	unknown_2:>
	unknown_3:>>
	unknown_4:>
	unknown_5:>>
	unknown_6:>
	unknown_7:>>
	unknown_8:>
	unknown_9:>>

unknown_10:>

unknown_11:>>

unknown_12:>

unknown_13:>>

unknown_14:>

unknown_15:>>

unknown_16:>

unknown_17:>>

unknown_18:>

unknown_19:>>

unknown_20:>

unknown_21:>>

unknown_22:>

unknown_23:>>

unknown_24:>

unknown_25:>>

unknown_26:>

unknown_27:>>

unknown_28:>

unknown_29:>

unknown_30:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30*,
Tin%
#2!*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*B
_read_only_resource_inputs$
" 	
 *-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_139899812
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_13991777

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_2__to__m_3/kernel/Regularizer/Abs?
%m_2__to__m_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_2__to__m_3/kernel/Regularizer/Const?
#m_2__to__m_3/kernel/Regularizer/SumSum'm_2__to__m_3/kernel/Regularizer/Abs:y:0.m_2__to__m_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/Sum?
%m_2__to__m_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_2__to__m_3/kernel/Regularizer/mul/x?
#m_2__to__m_3/kernel/Regularizer/mulMul.m_2__to__m_3/kernel/Regularizer/mul/x:output:0,m_2__to__m_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_13991745

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_1__to__m_2/kernel/Regularizer/Abs?
%m_1__to__m_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_1__to__m_2/kernel/Regularizer/Const?
#m_1__to__m_2/kernel/Regularizer/SumSum'm_1__to__m_2/kernel/Regularizer/Abs:y:0.m_1__to__m_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/Sum?
%m_1__to__m_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_1__to__m_2/kernel/Regularizer/mul/x?
#m_1__to__m_2/kernel/Regularizer/mulMul.m_1__to__m_2/kernel/Regularizer/mul/x:output:0,m_1__to__m_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
__inference_loss_fn_10_13992323O
=m_10__to__m_11_kernel_regularizer_abs_readvariableop_resource:>>
identity??4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_10__to__m_11_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_10__to__m_11/kernel/Regularizer/Abs?
'm_10__to__m_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_10__to__m_11/kernel/Regularizer/Const?
%m_10__to__m_11/kernel/Regularizer/SumSum)m_10__to__m_11/kernel/Regularizer/Abs:y:00m_10__to__m_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/Sum?
'm_10__to__m_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_10__to__m_11/kernel/Regularizer/mul/x?
%m_10__to__m_11/kernel/Regularizer/mulMul0m_10__to__m_11/kernel/Regularizer/mul/x:output:0.m_10__to__m_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/mul?
IdentityIdentity)m_10__to__m_11/kernel/Regularizer/mul:z:05^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp
?
?
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_13989694

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_7__to__m_8/kernel/Regularizer/Abs?
%m_7__to__m_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_7__to__m_8/kernel/Regularizer/Const?
#m_7__to__m_8/kernel/Regularizer/SumSum'm_7__to__m_8/kernel/Regularizer/Abs:y:0.m_7__to__m_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/Sum?
%m_7__to__m_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_7__to__m_8/kernel/Regularizer/mul/x?
#m_7__to__m_8/kernel/Regularizer/mulMul.m_7__to__m_8/kernel/Regularizer/mul/x:output:0,m_7__to__m_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
1__inference_m_12__to__m_13_layer_call_fn_13992106

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_139898092
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_13991713

inputs0
matmul_readvariableop_resource:>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2%
#m_0__to__m_1/kernel/Regularizer/Abs?
%m_0__to__m_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_0__to__m_1/kernel/Regularizer/Const?
#m_0__to__m_1/kernel/Regularizer/SumSum'm_0__to__m_1/kernel/Regularizer/Abs:y:0.m_0__to__m_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/Sum?
%m_0__to__m_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_0__to__m_1/kernel/Regularizer/mul/x?
#m_0__to__m_1/kernel/Regularizer/mulMul.m_0__to__m_1/kernel/Regularizer/mul/x:output:0,m_0__to__m_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
1__inference_m_13__to__m_14_layer_call_fn_13992138

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_139898322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
1__inference_m_11__to__m_12_layer_call_fn_13992074

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_139897862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
__inference_loss_fn_5_13992268M
;m_5__to__m_6_kernel_regularizer_abs_readvariableop_resource:>>
identity??2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_5__to__m_6_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_5__to__m_6/kernel/Regularizer/Abs?
%m_5__to__m_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_5__to__m_6/kernel/Regularizer/Const?
#m_5__to__m_6/kernel/Regularizer/SumSum'm_5__to__m_6/kernel/Regularizer/Abs:y:0.m_5__to__m_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/Sum?
%m_5__to__m_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_5__to__m_6/kernel/Regularizer/mul/x?
#m_5__to__m_6/kernel/Regularizer/mulMul.m_5__to__m_6/kernel/Regularizer/mul/x:output:0,m_5__to__m_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/mul?
IdentityIdentity'm_5__to__m_6/kernel/Regularizer/mul:z:03^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp
?
?
1__inference_m_15__to__m_16_layer_call_fn_13992202

inputs
unknown:>
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_15__to__m_16_layer_call_and_return_conditional_losses_139898782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_13992235M
;m_2__to__m_3_kernel_regularizer_abs_readvariableop_resource:>>
identity??2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_2__to__m_3_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_2__to__m_3/kernel/Regularizer/Abs?
%m_2__to__m_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_2__to__m_3/kernel/Regularizer/Const?
#m_2__to__m_3/kernel/Regularizer/SumSum'm_2__to__m_3/kernel/Regularizer/Abs:y:0.m_2__to__m_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/Sum?
%m_2__to__m_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_2__to__m_3/kernel/Regularizer/mul/x?
#m_2__to__m_3/kernel/Regularizer/mulMul.m_2__to__m_3/kernel/Regularizer/mul/x:output:0,m_2__to__m_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/mul?
IdentityIdentity'm_2__to__m_3/kernel/Regularizer/mul:z:03^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp
?
?
/__inference_m_7__to__m_8_layer_call_fn_13991946

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_139896942
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
L__inference_m_15__to__m_16_layer_call_and_return_conditional_losses_13992193

inputs0
matmul_readvariableop_resource:>-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd\
SoftmaxSoftmaxBiasAdd:z:0*
T0*'
_output_shapes
:?????????2	
Softmax?
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>*
dtype026
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
%m_15__to__m_16/kernel/Regularizer/AbsAbs<m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2'
%m_15__to__m_16/kernel/Regularizer/Abs?
'm_15__to__m_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_15__to__m_16/kernel/Regularizer/Const?
%m_15__to__m_16/kernel/Regularizer/SumSum)m_15__to__m_16/kernel/Regularizer/Abs:y:00m_15__to__m_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/Sum?
'm_15__to__m_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_15__to__m_16/kernel/Regularizer/mul/x?
%m_15__to__m_16/kernel/Regularizer/mulMul0m_15__to__m_16/kernel/Regularizer/mul/x:output:0.m_15__to__m_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/mul?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_13989533

inputs0
matmul_readvariableop_resource:>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2%
#m_0__to__m_1/kernel/Regularizer/Abs?
%m_0__to__m_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_0__to__m_1/kernel/Regularizer/Const?
#m_0__to__m_1/kernel/Regularizer/SumSum'm_0__to__m_1/kernel/Regularizer/Abs:y:0.m_0__to__m_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/Sum?
%m_0__to__m_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_0__to__m_1/kernel/Regularizer/mul/x?
#m_0__to__m_1/kernel/Regularizer/mulMul.m_0__to__m_1/kernel/Regularizer/mul/x:output:0,m_0__to__m_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_13992033

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_10__to__m_11/kernel/Regularizer/Abs?
'm_10__to__m_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_10__to__m_11/kernel/Regularizer/Const?
%m_10__to__m_11/kernel/Regularizer/SumSum)m_10__to__m_11/kernel/Regularizer/Abs:y:00m_10__to__m_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/Sum?
'm_10__to__m_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_10__to__m_11/kernel/Regularizer/mul/x?
%m_10__to__m_11/kernel/Regularizer/mulMul0m_10__to__m_11/kernel/Regularizer/mul/x:output:0.m_10__to__m_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_13989855

inputs0
matmul_readvariableop_resource:>>-
biasadd_readvariableop_resource:>
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:>*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
Relu?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_14__to__m_15/kernel/Regularizer/Abs?
'm_14__to__m_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_14__to__m_15/kernel/Regularizer/Const?
%m_14__to__m_15/kernel/Regularizer/SumSum)m_14__to__m_15/kernel/Regularizer/Abs:y:00m_14__to__m_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/Sum?
'm_14__to__m_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_14__to__m_15/kernel/Regularizer/mul/x?
%m_14__to__m_15/kernel/Regularizer/mulMul0m_14__to__m_15/kernel/Regularizer/mul/x:output:0.m_14__to__m_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/mul?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
??
? 
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13991552

inputs=
+m_0__to__m_1_matmul_readvariableop_resource:>:
,m_0__to__m_1_biasadd_readvariableop_resource:>=
+m_1__to__m_2_matmul_readvariableop_resource:>>:
,m_1__to__m_2_biasadd_readvariableop_resource:>=
+m_2__to__m_3_matmul_readvariableop_resource:>>:
,m_2__to__m_3_biasadd_readvariableop_resource:>=
+m_3__to__m_4_matmul_readvariableop_resource:>>:
,m_3__to__m_4_biasadd_readvariableop_resource:>=
+m_4__to__m_5_matmul_readvariableop_resource:>>:
,m_4__to__m_5_biasadd_readvariableop_resource:>=
+m_5__to__m_6_matmul_readvariableop_resource:>>:
,m_5__to__m_6_biasadd_readvariableop_resource:>=
+m_6__to__m_7_matmul_readvariableop_resource:>>:
,m_6__to__m_7_biasadd_readvariableop_resource:>=
+m_7__to__m_8_matmul_readvariableop_resource:>>:
,m_7__to__m_8_biasadd_readvariableop_resource:>=
+m_8__to__m_9_matmul_readvariableop_resource:>>:
,m_8__to__m_9_biasadd_readvariableop_resource:>>
,m_9__to__m_10_matmul_readvariableop_resource:>>;
-m_9__to__m_10_biasadd_readvariableop_resource:>?
-m_10__to__m_11_matmul_readvariableop_resource:>><
.m_10__to__m_11_biasadd_readvariableop_resource:>?
-m_11__to__m_12_matmul_readvariableop_resource:>><
.m_11__to__m_12_biasadd_readvariableop_resource:>?
-m_12__to__m_13_matmul_readvariableop_resource:>><
.m_12__to__m_13_biasadd_readvariableop_resource:>?
-m_13__to__m_14_matmul_readvariableop_resource:>><
.m_13__to__m_14_biasadd_readvariableop_resource:>?
-m_14__to__m_15_matmul_readvariableop_resource:>><
.m_14__to__m_15_biasadd_readvariableop_resource:>?
-m_15__to__m_16_matmul_readvariableop_resource:><
.m_15__to__m_16_biasadd_readvariableop_resource:
identity??#m_0__to__m_1/BiasAdd/ReadVariableOp?"m_0__to__m_1/MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?%m_10__to__m_11/BiasAdd/ReadVariableOp?$m_10__to__m_11/MatMul/ReadVariableOp?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?%m_11__to__m_12/BiasAdd/ReadVariableOp?$m_11__to__m_12/MatMul/ReadVariableOp?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?%m_12__to__m_13/BiasAdd/ReadVariableOp?$m_12__to__m_13/MatMul/ReadVariableOp?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?%m_13__to__m_14/BiasAdd/ReadVariableOp?$m_13__to__m_14/MatMul/ReadVariableOp?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?%m_14__to__m_15/BiasAdd/ReadVariableOp?$m_14__to__m_15/MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?%m_15__to__m_16/BiasAdd/ReadVariableOp?$m_15__to__m_16/MatMul/ReadVariableOp?4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?#m_1__to__m_2/BiasAdd/ReadVariableOp?"m_1__to__m_2/MatMul/ReadVariableOp?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?#m_2__to__m_3/BiasAdd/ReadVariableOp?"m_2__to__m_3/MatMul/ReadVariableOp?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?#m_3__to__m_4/BiasAdd/ReadVariableOp?"m_3__to__m_4/MatMul/ReadVariableOp?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?#m_4__to__m_5/BiasAdd/ReadVariableOp?"m_4__to__m_5/MatMul/ReadVariableOp?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?#m_5__to__m_6/BiasAdd/ReadVariableOp?"m_5__to__m_6/MatMul/ReadVariableOp?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?#m_6__to__m_7/BiasAdd/ReadVariableOp?"m_6__to__m_7/MatMul/ReadVariableOp?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?#m_7__to__m_8/BiasAdd/ReadVariableOp?"m_7__to__m_8/MatMul/ReadVariableOp?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?#m_8__to__m_9/BiasAdd/ReadVariableOp?"m_8__to__m_9/MatMul/ReadVariableOp?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?$m_9__to__m_10/BiasAdd/ReadVariableOp?#m_9__to__m_10/MatMul/ReadVariableOp?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
"m_0__to__m_1/MatMul/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:>*
dtype02$
"m_0__to__m_1/MatMul/ReadVariableOp?
m_0__to__m_1/MatMulMatMulinputs*m_0__to__m_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_0__to__m_1/MatMul?
#m_0__to__m_1/BiasAdd/ReadVariableOpReadVariableOp,m_0__to__m_1_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_0__to__m_1/BiasAdd/ReadVariableOp?
m_0__to__m_1/BiasAddAddm_0__to__m_1/MatMul:product:0+m_0__to__m_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_0__to__m_1/BiasAddz
m_0__to__m_1/ReluRelum_0__to__m_1/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_0__to__m_1/Relu?
"m_1__to__m_2/MatMul/ReadVariableOpReadVariableOp+m_1__to__m_2_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_1__to__m_2/MatMul/ReadVariableOp?
m_1__to__m_2/MatMulMatMulm_0__to__m_1/Relu:activations:0*m_1__to__m_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_1__to__m_2/MatMul?
#m_1__to__m_2/BiasAdd/ReadVariableOpReadVariableOp,m_1__to__m_2_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_1__to__m_2/BiasAdd/ReadVariableOp?
m_1__to__m_2/BiasAddAddm_1__to__m_2/MatMul:product:0+m_1__to__m_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_1__to__m_2/BiasAddz
m_1__to__m_2/ReluRelum_1__to__m_2/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_1__to__m_2/Relu?
"m_2__to__m_3/MatMul/ReadVariableOpReadVariableOp+m_2__to__m_3_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_2__to__m_3/MatMul/ReadVariableOp?
m_2__to__m_3/MatMulMatMulm_1__to__m_2/Relu:activations:0*m_2__to__m_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_2__to__m_3/MatMul?
#m_2__to__m_3/BiasAdd/ReadVariableOpReadVariableOp,m_2__to__m_3_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_2__to__m_3/BiasAdd/ReadVariableOp?
m_2__to__m_3/BiasAddAddm_2__to__m_3/MatMul:product:0+m_2__to__m_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_2__to__m_3/BiasAddz
m_2__to__m_3/ReluRelum_2__to__m_3/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_2__to__m_3/Relu?
"m_3__to__m_4/MatMul/ReadVariableOpReadVariableOp+m_3__to__m_4_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_3__to__m_4/MatMul/ReadVariableOp?
m_3__to__m_4/MatMulMatMulm_2__to__m_3/Relu:activations:0*m_3__to__m_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_3__to__m_4/MatMul?
#m_3__to__m_4/BiasAdd/ReadVariableOpReadVariableOp,m_3__to__m_4_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_3__to__m_4/BiasAdd/ReadVariableOp?
m_3__to__m_4/BiasAddAddm_3__to__m_4/MatMul:product:0+m_3__to__m_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_3__to__m_4/BiasAddz
m_3__to__m_4/ReluRelum_3__to__m_4/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_3__to__m_4/Relu?
"m_4__to__m_5/MatMul/ReadVariableOpReadVariableOp+m_4__to__m_5_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_4__to__m_5/MatMul/ReadVariableOp?
m_4__to__m_5/MatMulMatMulm_3__to__m_4/Relu:activations:0*m_4__to__m_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_4__to__m_5/MatMul?
#m_4__to__m_5/BiasAdd/ReadVariableOpReadVariableOp,m_4__to__m_5_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_4__to__m_5/BiasAdd/ReadVariableOp?
m_4__to__m_5/BiasAddAddm_4__to__m_5/MatMul:product:0+m_4__to__m_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_4__to__m_5/BiasAddz
m_4__to__m_5/ReluRelum_4__to__m_5/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_4__to__m_5/Relu?
"m_5__to__m_6/MatMul/ReadVariableOpReadVariableOp+m_5__to__m_6_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_5__to__m_6/MatMul/ReadVariableOp?
m_5__to__m_6/MatMulMatMulm_4__to__m_5/Relu:activations:0*m_5__to__m_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_5__to__m_6/MatMul?
#m_5__to__m_6/BiasAdd/ReadVariableOpReadVariableOp,m_5__to__m_6_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_5__to__m_6/BiasAdd/ReadVariableOp?
m_5__to__m_6/BiasAddAddm_5__to__m_6/MatMul:product:0+m_5__to__m_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_5__to__m_6/BiasAddz
m_5__to__m_6/ReluRelum_5__to__m_6/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_5__to__m_6/Relu?
"m_6__to__m_7/MatMul/ReadVariableOpReadVariableOp+m_6__to__m_7_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_6__to__m_7/MatMul/ReadVariableOp?
m_6__to__m_7/MatMulMatMulm_5__to__m_6/Relu:activations:0*m_6__to__m_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_6__to__m_7/MatMul?
#m_6__to__m_7/BiasAdd/ReadVariableOpReadVariableOp,m_6__to__m_7_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_6__to__m_7/BiasAdd/ReadVariableOp?
m_6__to__m_7/BiasAddAddm_6__to__m_7/MatMul:product:0+m_6__to__m_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_6__to__m_7/BiasAddz
m_6__to__m_7/ReluRelum_6__to__m_7/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_6__to__m_7/Relu?
"m_7__to__m_8/MatMul/ReadVariableOpReadVariableOp+m_7__to__m_8_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_7__to__m_8/MatMul/ReadVariableOp?
m_7__to__m_8/MatMulMatMulm_6__to__m_7/Relu:activations:0*m_7__to__m_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_7__to__m_8/MatMul?
#m_7__to__m_8/BiasAdd/ReadVariableOpReadVariableOp,m_7__to__m_8_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_7__to__m_8/BiasAdd/ReadVariableOp?
m_7__to__m_8/BiasAddAddm_7__to__m_8/MatMul:product:0+m_7__to__m_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_7__to__m_8/BiasAddz
m_7__to__m_8/ReluRelum_7__to__m_8/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_7__to__m_8/Relu?
"m_8__to__m_9/MatMul/ReadVariableOpReadVariableOp+m_8__to__m_9_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02$
"m_8__to__m_9/MatMul/ReadVariableOp?
m_8__to__m_9/MatMulMatMulm_7__to__m_8/Relu:activations:0*m_8__to__m_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_8__to__m_9/MatMul?
#m_8__to__m_9/BiasAdd/ReadVariableOpReadVariableOp,m_8__to__m_9_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02%
#m_8__to__m_9/BiasAdd/ReadVariableOp?
m_8__to__m_9/BiasAddAddm_8__to__m_9/MatMul:product:0+m_8__to__m_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_8__to__m_9/BiasAddz
m_8__to__m_9/ReluRelum_8__to__m_9/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_8__to__m_9/Relu?
#m_9__to__m_10/MatMul/ReadVariableOpReadVariableOp,m_9__to__m_10_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02%
#m_9__to__m_10/MatMul/ReadVariableOp?
m_9__to__m_10/MatMulMatMulm_8__to__m_9/Relu:activations:0+m_9__to__m_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_9__to__m_10/MatMul?
$m_9__to__m_10/BiasAdd/ReadVariableOpReadVariableOp-m_9__to__m_10_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02&
$m_9__to__m_10/BiasAdd/ReadVariableOp?
m_9__to__m_10/BiasAddAddm_9__to__m_10/MatMul:product:0,m_9__to__m_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_9__to__m_10/BiasAdd}
m_9__to__m_10/ReluRelum_9__to__m_10/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_9__to__m_10/Relu?
$m_10__to__m_11/MatMul/ReadVariableOpReadVariableOp-m_10__to__m_11_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02&
$m_10__to__m_11/MatMul/ReadVariableOp?
m_10__to__m_11/MatMulMatMul m_9__to__m_10/Relu:activations:0,m_10__to__m_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_10__to__m_11/MatMul?
%m_10__to__m_11/BiasAdd/ReadVariableOpReadVariableOp.m_10__to__m_11_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02'
%m_10__to__m_11/BiasAdd/ReadVariableOp?
m_10__to__m_11/BiasAddAddm_10__to__m_11/MatMul:product:0-m_10__to__m_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_10__to__m_11/BiasAdd?
m_10__to__m_11/ReluRelum_10__to__m_11/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_10__to__m_11/Relu?
$m_11__to__m_12/MatMul/ReadVariableOpReadVariableOp-m_11__to__m_12_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02&
$m_11__to__m_12/MatMul/ReadVariableOp?
m_11__to__m_12/MatMulMatMul!m_10__to__m_11/Relu:activations:0,m_11__to__m_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_11__to__m_12/MatMul?
%m_11__to__m_12/BiasAdd/ReadVariableOpReadVariableOp.m_11__to__m_12_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02'
%m_11__to__m_12/BiasAdd/ReadVariableOp?
m_11__to__m_12/BiasAddAddm_11__to__m_12/MatMul:product:0-m_11__to__m_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_11__to__m_12/BiasAdd?
m_11__to__m_12/ReluRelum_11__to__m_12/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_11__to__m_12/Relu?
$m_12__to__m_13/MatMul/ReadVariableOpReadVariableOp-m_12__to__m_13_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02&
$m_12__to__m_13/MatMul/ReadVariableOp?
m_12__to__m_13/MatMulMatMul!m_11__to__m_12/Relu:activations:0,m_12__to__m_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_12__to__m_13/MatMul?
%m_12__to__m_13/BiasAdd/ReadVariableOpReadVariableOp.m_12__to__m_13_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02'
%m_12__to__m_13/BiasAdd/ReadVariableOp?
m_12__to__m_13/BiasAddAddm_12__to__m_13/MatMul:product:0-m_12__to__m_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_12__to__m_13/BiasAdd?
m_12__to__m_13/ReluRelum_12__to__m_13/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_12__to__m_13/Relu?
$m_13__to__m_14/MatMul/ReadVariableOpReadVariableOp-m_13__to__m_14_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02&
$m_13__to__m_14/MatMul/ReadVariableOp?
m_13__to__m_14/MatMulMatMul!m_12__to__m_13/Relu:activations:0,m_13__to__m_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_13__to__m_14/MatMul?
%m_13__to__m_14/BiasAdd/ReadVariableOpReadVariableOp.m_13__to__m_14_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02'
%m_13__to__m_14/BiasAdd/ReadVariableOp?
m_13__to__m_14/BiasAddAddm_13__to__m_14/MatMul:product:0-m_13__to__m_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_13__to__m_14/BiasAdd?
m_13__to__m_14/ReluRelum_13__to__m_14/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_13__to__m_14/Relu?
$m_14__to__m_15/MatMul/ReadVariableOpReadVariableOp-m_14__to__m_15_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype02&
$m_14__to__m_15/MatMul/ReadVariableOp?
m_14__to__m_15/MatMulMatMul!m_13__to__m_14/Relu:activations:0,m_14__to__m_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_14__to__m_15/MatMul?
%m_14__to__m_15/BiasAdd/ReadVariableOpReadVariableOp.m_14__to__m_15_biasadd_readvariableop_resource*
_output_shapes
:>*
dtype02'
%m_14__to__m_15/BiasAdd/ReadVariableOp?
m_14__to__m_15/BiasAddAddm_14__to__m_15/MatMul:product:0-m_14__to__m_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????>2
m_14__to__m_15/BiasAdd?
m_14__to__m_15/ReluRelum_14__to__m_15/BiasAdd:z:0*
T0*'
_output_shapes
:?????????>2
m_14__to__m_15/Relu?
$m_15__to__m_16/MatMul/ReadVariableOpReadVariableOp-m_15__to__m_16_matmul_readvariableop_resource*
_output_shapes

:>*
dtype02&
$m_15__to__m_16/MatMul/ReadVariableOp?
m_15__to__m_16/MatMulMatMul!m_14__to__m_15/Relu:activations:0,m_15__to__m_16/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_15__to__m_16/MatMul?
%m_15__to__m_16/BiasAdd/ReadVariableOpReadVariableOp.m_15__to__m_16_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%m_15__to__m_16/BiasAdd/ReadVariableOp?
m_15__to__m_16/BiasAddAddm_15__to__m_16/MatMul:product:0-m_15__to__m_16/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_15__to__m_16/BiasAdd?
m_15__to__m_16/SoftmaxSoftmaxm_15__to__m_16/BiasAdd:z:0*
T0*'
_output_shapes
:?????????2
m_15__to__m_16/Softmax?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:>*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2%
#m_0__to__m_1/kernel/Regularizer/Abs?
%m_0__to__m_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_0__to__m_1/kernel/Regularizer/Const?
#m_0__to__m_1/kernel/Regularizer/SumSum'm_0__to__m_1/kernel/Regularizer/Abs:y:0.m_0__to__m_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/Sum?
%m_0__to__m_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_0__to__m_1/kernel/Regularizer/mul/x?
#m_0__to__m_1/kernel/Regularizer/mulMul.m_0__to__m_1/kernel/Regularizer/mul/x:output:0,m_0__to__m_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_0__to__m_1/kernel/Regularizer/mul?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_1__to__m_2_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_1__to__m_2/kernel/Regularizer/Abs?
%m_1__to__m_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_1__to__m_2/kernel/Regularizer/Const?
#m_1__to__m_2/kernel/Regularizer/SumSum'm_1__to__m_2/kernel/Regularizer/Abs:y:0.m_1__to__m_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/Sum?
%m_1__to__m_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_1__to__m_2/kernel/Regularizer/mul/x?
#m_1__to__m_2/kernel/Regularizer/mulMul.m_1__to__m_2/kernel/Regularizer/mul/x:output:0,m_1__to__m_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_1__to__m_2/kernel/Regularizer/mul?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_2__to__m_3_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_2__to__m_3/kernel/Regularizer/Abs?
%m_2__to__m_3/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_2__to__m_3/kernel/Regularizer/Const?
#m_2__to__m_3/kernel/Regularizer/SumSum'm_2__to__m_3/kernel/Regularizer/Abs:y:0.m_2__to__m_3/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/Sum?
%m_2__to__m_3/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_2__to__m_3/kernel/Regularizer/mul/x?
#m_2__to__m_3/kernel/Regularizer/mulMul.m_2__to__m_3/kernel/Regularizer/mul/x:output:0,m_2__to__m_3/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_2__to__m_3/kernel/Regularizer/mul?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_3__to__m_4_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_3__to__m_4/kernel/Regularizer/Abs?
%m_3__to__m_4/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_3__to__m_4/kernel/Regularizer/Const?
#m_3__to__m_4/kernel/Regularizer/SumSum'm_3__to__m_4/kernel/Regularizer/Abs:y:0.m_3__to__m_4/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/Sum?
%m_3__to__m_4/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_3__to__m_4/kernel/Regularizer/mul/x?
#m_3__to__m_4/kernel/Regularizer/mulMul.m_3__to__m_4/kernel/Regularizer/mul/x:output:0,m_3__to__m_4/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_3__to__m_4/kernel/Regularizer/mul?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_4__to__m_5_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_4__to__m_5/kernel/Regularizer/Abs?
%m_4__to__m_5/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_4__to__m_5/kernel/Regularizer/Const?
#m_4__to__m_5/kernel/Regularizer/SumSum'm_4__to__m_5/kernel/Regularizer/Abs:y:0.m_4__to__m_5/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/Sum?
%m_4__to__m_5/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_4__to__m_5/kernel/Regularizer/mul/x?
#m_4__to__m_5/kernel/Regularizer/mulMul.m_4__to__m_5/kernel/Regularizer/mul/x:output:0,m_4__to__m_5/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_4__to__m_5/kernel/Regularizer/mul?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_5__to__m_6_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_5__to__m_6/kernel/Regularizer/Abs?
%m_5__to__m_6/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_5__to__m_6/kernel/Regularizer/Const?
#m_5__to__m_6/kernel/Regularizer/SumSum'm_5__to__m_6/kernel/Regularizer/Abs:y:0.m_5__to__m_6/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/Sum?
%m_5__to__m_6/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_5__to__m_6/kernel/Regularizer/mul/x?
#m_5__to__m_6/kernel/Regularizer/mulMul.m_5__to__m_6/kernel/Regularizer/mul/x:output:0,m_5__to__m_6/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_5__to__m_6/kernel/Regularizer/mul?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_6__to__m_7_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_6__to__m_7/kernel/Regularizer/Abs?
%m_6__to__m_7/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_6__to__m_7/kernel/Regularizer/Const?
#m_6__to__m_7/kernel/Regularizer/SumSum'm_6__to__m_7/kernel/Regularizer/Abs:y:0.m_6__to__m_7/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/Sum?
%m_6__to__m_7/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_6__to__m_7/kernel/Regularizer/mul/x?
#m_6__to__m_7/kernel/Regularizer/mulMul.m_6__to__m_7/kernel/Regularizer/mul/x:output:0,m_6__to__m_7/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_6__to__m_7/kernel/Regularizer/mul?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_7__to__m_8_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_7__to__m_8/kernel/Regularizer/Abs?
%m_7__to__m_8/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_7__to__m_8/kernel/Regularizer/Const?
#m_7__to__m_8/kernel/Regularizer/SumSum'm_7__to__m_8/kernel/Regularizer/Abs:y:0.m_7__to__m_8/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/Sum?
%m_7__to__m_8/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_7__to__m_8/kernel/Regularizer/mul/x?
#m_7__to__m_8/kernel/Regularizer/mulMul.m_7__to__m_8/kernel/Regularizer/mul/x:output:0,m_7__to__m_8/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_7__to__m_8/kernel/Regularizer/mul?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_8__to__m_9_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2%
#m_8__to__m_9/kernel/Regularizer/Abs?
%m_8__to__m_9/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2'
%m_8__to__m_9/kernel/Regularizer/Const?
#m_8__to__m_9/kernel/Regularizer/SumSum'm_8__to__m_9/kernel/Regularizer/Abs:y:0.m_8__to__m_9/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/Sum?
%m_8__to__m_9/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92'
%m_8__to__m_9/kernel/Regularizer/mul/x?
#m_8__to__m_9/kernel/Regularizer/mulMul.m_8__to__m_9/kernel/Regularizer/mul/x:output:0,m_8__to__m_9/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2%
#m_8__to__m_9/kernel/Regularizer/mul?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp,m_9__to__m_10_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2&
$m_9__to__m_10/kernel/Regularizer/Abs?
&m_9__to__m_10/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2(
&m_9__to__m_10/kernel/Regularizer/Const?
$m_9__to__m_10/kernel/Regularizer/SumSum(m_9__to__m_10/kernel/Regularizer/Abs:y:0/m_9__to__m_10/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/Sum?
&m_9__to__m_10/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92(
&m_9__to__m_10/kernel/Regularizer/mul/x?
$m_9__to__m_10/kernel/Regularizer/mulMul/m_9__to__m_10/kernel/Regularizer/mul/x:output:0-m_9__to__m_10/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2&
$m_9__to__m_10/kernel/Regularizer/mul?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_10__to__m_11_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_10__to__m_11/kernel/Regularizer/Abs?
'm_10__to__m_11/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_10__to__m_11/kernel/Regularizer/Const?
%m_10__to__m_11/kernel/Regularizer/SumSum)m_10__to__m_11/kernel/Regularizer/Abs:y:00m_10__to__m_11/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/Sum?
'm_10__to__m_11/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_10__to__m_11/kernel/Regularizer/mul/x?
%m_10__to__m_11/kernel/Regularizer/mulMul0m_10__to__m_11/kernel/Regularizer/mul/x:output:0.m_10__to__m_11/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_10__to__m_11/kernel/Regularizer/mul?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_11__to__m_12_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_11__to__m_12/kernel/Regularizer/Abs?
'm_11__to__m_12/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_11__to__m_12/kernel/Regularizer/Const?
%m_11__to__m_12/kernel/Regularizer/SumSum)m_11__to__m_12/kernel/Regularizer/Abs:y:00m_11__to__m_12/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/Sum?
'm_11__to__m_12/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_11__to__m_12/kernel/Regularizer/mul/x?
%m_11__to__m_12/kernel/Regularizer/mulMul0m_11__to__m_12/kernel/Regularizer/mul/x:output:0.m_11__to__m_12/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_11__to__m_12/kernel/Regularizer/mul?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_12__to__m_13_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_12__to__m_13/kernel/Regularizer/Abs?
'm_12__to__m_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_12__to__m_13/kernel/Regularizer/Const?
%m_12__to__m_13/kernel/Regularizer/SumSum)m_12__to__m_13/kernel/Regularizer/Abs:y:00m_12__to__m_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/Sum?
'm_12__to__m_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_12__to__m_13/kernel/Regularizer/mul/x?
%m_12__to__m_13/kernel/Regularizer/mulMul0m_12__to__m_13/kernel/Regularizer/mul/x:output:0.m_12__to__m_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/mul?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_13__to__m_14_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_13__to__m_14/kernel/Regularizer/Abs?
'm_13__to__m_14/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_13__to__m_14/kernel/Regularizer/Const?
%m_13__to__m_14/kernel/Regularizer/SumSum)m_13__to__m_14/kernel/Regularizer/Abs:y:00m_13__to__m_14/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/Sum?
'm_13__to__m_14/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_13__to__m_14/kernel/Regularizer/mul/x?
%m_13__to__m_14/kernel/Regularizer/mulMul0m_13__to__m_14/kernel/Regularizer/mul/x:output:0.m_13__to__m_14/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_13__to__m_14/kernel/Regularizer/mul?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_14__to__m_15_matmul_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_14__to__m_15/kernel/Regularizer/Abs?
'm_14__to__m_15/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_14__to__m_15/kernel/Regularizer/Const?
%m_14__to__m_15/kernel/Regularizer/SumSum)m_14__to__m_15/kernel/Regularizer/Abs:y:00m_14__to__m_15/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/Sum?
'm_14__to__m_15/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_14__to__m_15/kernel/Regularizer/mul/x?
%m_14__to__m_15/kernel/Regularizer/mulMul0m_14__to__m_15/kernel/Regularizer/mul/x:output:0.m_14__to__m_15/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_14__to__m_15/kernel/Regularizer/mul?
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp-m_15__to__m_16_matmul_readvariableop_resource*
_output_shapes

:>*
dtype026
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp?
%m_15__to__m_16/kernel/Regularizer/AbsAbs<m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>2'
%m_15__to__m_16/kernel/Regularizer/Abs?
'm_15__to__m_16/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_15__to__m_16/kernel/Regularizer/Const?
%m_15__to__m_16/kernel/Regularizer/SumSum)m_15__to__m_16/kernel/Regularizer/Abs:y:00m_15__to__m_16/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/Sum?
'm_15__to__m_16/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_15__to__m_16/kernel/Regularizer/mul/x?
%m_15__to__m_16/kernel/Regularizer/mulMul0m_15__to__m_16/kernel/Regularizer/mul/x:output:0.m_15__to__m_16/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_15__to__m_16/kernel/Regularizer/mul?
IdentityIdentity m_15__to__m_16/Softmax:softmax:0$^m_0__to__m_1/BiasAdd/ReadVariableOp#^m_0__to__m_1/MatMul/ReadVariableOp3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp&^m_10__to__m_11/BiasAdd/ReadVariableOp%^m_10__to__m_11/MatMul/ReadVariableOp5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp&^m_11__to__m_12/BiasAdd/ReadVariableOp%^m_11__to__m_12/MatMul/ReadVariableOp5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp&^m_12__to__m_13/BiasAdd/ReadVariableOp%^m_12__to__m_13/MatMul/ReadVariableOp5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp&^m_13__to__m_14/BiasAdd/ReadVariableOp%^m_13__to__m_14/MatMul/ReadVariableOp5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp&^m_14__to__m_15/BiasAdd/ReadVariableOp%^m_14__to__m_15/MatMul/ReadVariableOp5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp&^m_15__to__m_16/BiasAdd/ReadVariableOp%^m_15__to__m_16/MatMul/ReadVariableOp5^m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp$^m_1__to__m_2/BiasAdd/ReadVariableOp#^m_1__to__m_2/MatMul/ReadVariableOp3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp$^m_2__to__m_3/BiasAdd/ReadVariableOp#^m_2__to__m_3/MatMul/ReadVariableOp3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp$^m_3__to__m_4/BiasAdd/ReadVariableOp#^m_3__to__m_4/MatMul/ReadVariableOp3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp$^m_4__to__m_5/BiasAdd/ReadVariableOp#^m_4__to__m_5/MatMul/ReadVariableOp3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp$^m_5__to__m_6/BiasAdd/ReadVariableOp#^m_5__to__m_6/MatMul/ReadVariableOp3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp$^m_6__to__m_7/BiasAdd/ReadVariableOp#^m_6__to__m_7/MatMul/ReadVariableOp3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp$^m_7__to__m_8/BiasAdd/ReadVariableOp#^m_7__to__m_8/MatMul/ReadVariableOp3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp$^m_8__to__m_9/BiasAdd/ReadVariableOp#^m_8__to__m_9/MatMul/ReadVariableOp3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp%^m_9__to__m_10/BiasAdd/ReadVariableOp$^m_9__to__m_10/MatMul/ReadVariableOp4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*f
_input_shapesU
S:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
#m_0__to__m_1/BiasAdd/ReadVariableOp#m_0__to__m_1/BiasAdd/ReadVariableOp2H
"m_0__to__m_1/MatMul/ReadVariableOp"m_0__to__m_1/MatMul/ReadVariableOp2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2N
%m_10__to__m_11/BiasAdd/ReadVariableOp%m_10__to__m_11/BiasAdd/ReadVariableOp2L
$m_10__to__m_11/MatMul/ReadVariableOp$m_10__to__m_11/MatMul/ReadVariableOp2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp2N
%m_11__to__m_12/BiasAdd/ReadVariableOp%m_11__to__m_12/BiasAdd/ReadVariableOp2L
$m_11__to__m_12/MatMul/ReadVariableOp$m_11__to__m_12/MatMul/ReadVariableOp2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp2N
%m_12__to__m_13/BiasAdd/ReadVariableOp%m_12__to__m_13/BiasAdd/ReadVariableOp2L
$m_12__to__m_13/MatMul/ReadVariableOp$m_12__to__m_13/MatMul/ReadVariableOp2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp2N
%m_13__to__m_14/BiasAdd/ReadVariableOp%m_13__to__m_14/BiasAdd/ReadVariableOp2L
$m_13__to__m_14/MatMul/ReadVariableOp$m_13__to__m_14/MatMul/ReadVariableOp2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp2N
%m_14__to__m_15/BiasAdd/ReadVariableOp%m_14__to__m_15/BiasAdd/ReadVariableOp2L
$m_14__to__m_15/MatMul/ReadVariableOp$m_14__to__m_15/MatMul/ReadVariableOp2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2N
%m_15__to__m_16/BiasAdd/ReadVariableOp%m_15__to__m_16/BiasAdd/ReadVariableOp2L
$m_15__to__m_16/MatMul/ReadVariableOp$m_15__to__m_16/MatMul/ReadVariableOp2l
4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp4m_15__to__m_16/kernel/Regularizer/Abs/ReadVariableOp2J
#m_1__to__m_2/BiasAdd/ReadVariableOp#m_1__to__m_2/BiasAdd/ReadVariableOp2H
"m_1__to__m_2/MatMul/ReadVariableOp"m_1__to__m_2/MatMul/ReadVariableOp2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2J
#m_2__to__m_3/BiasAdd/ReadVariableOp#m_2__to__m_3/BiasAdd/ReadVariableOp2H
"m_2__to__m_3/MatMul/ReadVariableOp"m_2__to__m_3/MatMul/ReadVariableOp2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2J
#m_3__to__m_4/BiasAdd/ReadVariableOp#m_3__to__m_4/BiasAdd/ReadVariableOp2H
"m_3__to__m_4/MatMul/ReadVariableOp"m_3__to__m_4/MatMul/ReadVariableOp2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2J
#m_4__to__m_5/BiasAdd/ReadVariableOp#m_4__to__m_5/BiasAdd/ReadVariableOp2H
"m_4__to__m_5/MatMul/ReadVariableOp"m_4__to__m_5/MatMul/ReadVariableOp2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2J
#m_5__to__m_6/BiasAdd/ReadVariableOp#m_5__to__m_6/BiasAdd/ReadVariableOp2H
"m_5__to__m_6/MatMul/ReadVariableOp"m_5__to__m_6/MatMul/ReadVariableOp2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2J
#m_6__to__m_7/BiasAdd/ReadVariableOp#m_6__to__m_7/BiasAdd/ReadVariableOp2H
"m_6__to__m_7/MatMul/ReadVariableOp"m_6__to__m_7/MatMul/ReadVariableOp2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2J
#m_7__to__m_8/BiasAdd/ReadVariableOp#m_7__to__m_8/BiasAdd/ReadVariableOp2H
"m_7__to__m_8/MatMul/ReadVariableOp"m_7__to__m_8/MatMul/ReadVariableOp2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2J
#m_8__to__m_9/BiasAdd/ReadVariableOp#m_8__to__m_9/BiasAdd/ReadVariableOp2H
"m_8__to__m_9/MatMul/ReadVariableOp"m_8__to__m_9/MatMul/ReadVariableOp2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2L
$m_9__to__m_10/BiasAdd/ReadVariableOp$m_9__to__m_10/BiasAdd/ReadVariableOp2J
#m_9__to__m_10/MatMul/ReadVariableOp#m_9__to__m_10/MatMul/ReadVariableOp2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_m_2__to__m_3_layer_call_fn_13991786

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_139895792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs
?
?
__inference_loss_fn_12_13992345O
=m_12__to__m_13_kernel_regularizer_abs_readvariableop_resource:>>
identity??4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_12__to__m_13_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:>>*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:>>2'
%m_12__to__m_13/kernel/Regularizer/Abs?
'm_12__to__m_13/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2)
'm_12__to__m_13/kernel/Regularizer/Const?
%m_12__to__m_13/kernel/Regularizer/SumSum)m_12__to__m_13/kernel/Regularizer/Abs:y:00m_12__to__m_13/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/Sum?
'm_12__to__m_13/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *?Q92)
'm_12__to__m_13/kernel/Regularizer/mul/x?
%m_12__to__m_13/kernel/Regularizer/mulMul0m_12__to__m_13/kernel/Regularizer/mul/x:output:0.m_12__to__m_13/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2'
%m_12__to__m_13/kernel/Regularizer/mul?
IdentityIdentity)m_12__to__m_13/kernel/Regularizer/mul:z:05^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: 2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp
?
?
/__inference_m_8__to__m_9_layer_call_fn_13991978

inputs
unknown:>>
	unknown_0:>
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????>*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_139897172
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????>2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????>: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????>
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
Q
m_0__to__m_1_input;
$serving_default_m_0__to__m_1_input:0?????????B
m_15__to__m_160
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
??
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
layer_with_weights-6
layer-6
layer_with_weights-7
layer-7
	layer_with_weights-8
	layer-8

layer_with_weights-9

layer-9
layer_with_weights-10
layer-10
layer_with_weights-11
layer-11
layer_with_weights-12
layer-12
layer_with_weights-13
layer-13
layer_with_weights-14
layer-14
layer_with_weights-15
layer-15
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"Ȓ
_tf_keras_sequential??{"name": "L-15-m_0-1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "L-15-m_0-1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "m_0__to__m_1_input"}}, {"class_name": "Dense", "config": {"name": "m_0__to__m_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_1__to__m_2", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_2__to__m_3", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_3__to__m_4", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_4__to__m_5", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_5__to__m_6", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_6__to__m_7", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_7__to__m_8", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_8__to__m_9", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_9__to__m_10", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_10__to__m_11", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_11__to__m_12", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_12__to__m_13", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_13__to__m_14", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_14__to__m_15", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_15__to__m_16", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 65, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "m_0__to__m_1_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "L-15-m_0-1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "m_0__to__m_1_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "m_0__to__m_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}, {"class_name": "Dense", "config": {"name": "m_1__to__m_2", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 7}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}, {"class_name": "Dense", "config": {"name": "m_2__to__m_3", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "m_3__to__m_4", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 15}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "Dense", "config": {"name": "m_4__to__m_5", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 19}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "Dense", "config": {"name": "m_5__to__m_6", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 23}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "Dense", "config": {"name": "m_6__to__m_7", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 27}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "m_7__to__m_8", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dense", "config": {"name": "m_8__to__m_9", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 35}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36}, {"class_name": "Dense", "config": {"name": "m_9__to__m_10", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 39}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40}, {"class_name": "Dense", "config": {"name": "m_10__to__m_11", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 43}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 44}, {"class_name": "Dense", "config": {"name": "m_11__to__m_12", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 47}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 48}, {"class_name": "Dense", "config": {"name": "m_12__to__m_13", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 51}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 52}, {"class_name": "Dense", "config": {"name": "m_13__to__m_14", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 55}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 56}, {"class_name": "Dense", "config": {"name": "m_14__to__m_15", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 59}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60}, {"class_name": "Dense", "config": {"name": "m_15__to__m_16", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 63}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 64}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0005000000237487257, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_0__to__m_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_0__to__m_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?	

kernel
bias
regularization_losses
 	variables
!trainable_variables
"	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_1__to__m_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_1__to__m_2", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 7}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

#kernel
$bias
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_2__to__m_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_2__to__m_3", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_3__to__m_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_3__to__m_4", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 15}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

/kernel
0bias
1regularization_losses
2	variables
3trainable_variables
4	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_4__to__m_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_4__to__m_5", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 19}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

5kernel
6bias
7regularization_losses
8	variables
9trainable_variables
:	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_5__to__m_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_5__to__m_6", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 23}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

;kernel
<bias
=regularization_losses
>	variables
?trainable_variables
@	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_6__to__m_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_6__to__m_7", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 27}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

Akernel
Bbias
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_7__to__m_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_7__to__m_8", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

Gkernel
Hbias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_8__to__m_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_8__to__m_9", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 35}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 74}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

Mkernel
Nbias
Oregularization_losses
P	variables
Qtrainable_variables
R	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_9__to__m_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_9__to__m_10", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 39}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

Skernel
Tbias
Uregularization_losses
V	variables
Wtrainable_variables
X	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_10__to__m_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_10__to__m_11", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 43}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 76}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

Ykernel
Zbias
[regularization_losses
\	variables
]trainable_variables
^	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_11__to__m_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_11__to__m_12", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 47}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 77}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

_kernel
`bias
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_12__to__m_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_12__to__m_13", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 51}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 78}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_13__to__m_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_13__to__m_14", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 55}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 56, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 79}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

kkernel
lbias
mregularization_losses
n	variables
otrainable_variables
p	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_14__to__m_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_14__to__m_15", "trainable": true, "dtype": "float32", "units": 62, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 59}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?	

qkernel
rbias
sregularization_losses
t	variables
utrainable_variables
v	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_15__to__m_16", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_15__to__m_16", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 63}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 64, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 62}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 62]}}
?
witer

xbeta_1

ybeta_2
	zdecay
{learning_ratem?m?m?m?#m?$m?)m?*m?/m?0m?5m?6m?;m?<m?Am?Bm?Gm?Hm?Mm?Nm?Sm?Tm?Ym?Zm?_m?`m?em?fm?km?lm?qm?rm?v?v?v?v?#v?$v?)v?*v?/v?0v?5v?6v?;v?<v?Av?Bv?Gv?Hv?Mv?Nv?Sv?Tv?Yv?Zv?_v?`v?ev?fv?kv?lv?qv?rv?"
	optimizer
?
?0
?1
?2
?3
?4
?5
?6
?7
?8
?9
?10
?11
?12
?13
?14
?15"
trackable_list_wrapper
?
0
1
2
3
#4
$5
)6
*7
/8
09
510
611
;12
<13
A14
B15
G16
H17
M18
N19
S20
T21
Y22
Z23
_24
`25
e26
f27
k28
l29
q30
r31"
trackable_list_wrapper
?
0
1
2
3
#4
$5
)6
*7
/8
09
510
611
;12
<13
A14
B15
G16
H17
M18
N19
S20
T21
Y22
Z23
_24
`25
e26
f27
k28
l29
q30
r31"
trackable_list_wrapper
?
regularization_losses
|layer_regularization_losses
}metrics
~non_trainable_variables

layers
	variables
trainable_variables
?layer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:#>2m_0__to__m_1/kernel
:>2m_0__to__m_1/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
	variables
trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#>>2m_1__to__m_2/kernel
:>2m_1__to__m_2/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
 	variables
!trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#>>2m_2__to__m_3/kernel
:>2m_2__to__m_3/bias
(
?0"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
?
%regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
&	variables
'trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#>>2m_3__to__m_4/kernel
:>2m_3__to__m_4/bias
(
?0"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
?
+regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
,	variables
-trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#>>2m_4__to__m_5/kernel
:>2m_4__to__m_5/bias
(
?0"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
?
1regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
2	variables
3trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#>>2m_5__to__m_6/kernel
:>2m_5__to__m_6/bias
(
?0"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
?
7regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
8	variables
9trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#>>2m_6__to__m_7/kernel
:>2m_6__to__m_7/bias
(
?0"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
?
=regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
>	variables
?trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#>>2m_7__to__m_8/kernel
:>2m_7__to__m_8/bias
(
?0"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
?
Cregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
D	variables
Etrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#>>2m_8__to__m_9/kernel
:>2m_8__to__m_9/bias
(
?0"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
?
Iregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
J	variables
Ktrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$>>2m_9__to__m_10/kernel
 :>2m_9__to__m_10/bias
(
?0"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
?
Oregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
P	variables
Qtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%>>2m_10__to__m_11/kernel
!:>2m_10__to__m_11/bias
(
?0"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
?
Uregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
V	variables
Wtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%>>2m_11__to__m_12/kernel
!:>2m_11__to__m_12/bias
(
?0"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
?
[regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
\	variables
]trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%>>2m_12__to__m_13/kernel
!:>2m_12__to__m_13/bias
(
?0"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
?
aregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
b	variables
ctrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%>>2m_13__to__m_14/kernel
!:>2m_13__to__m_14/bias
(
?0"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
?
gregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
h	variables
itrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%>>2m_14__to__m_15/kernel
!:>2m_14__to__m_15/bias
(
?0"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
?
mregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
n	variables
otrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%>2m_15__to__m_16/kernel
!:2m_15__to__m_16/bias
(
?0"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
.
q0
r1"
trackable_list_wrapper
?
sregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
t	variables
utrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15"
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
(
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 82}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(>2Adam/m_0__to__m_1/kernel/m
$:">2Adam/m_0__to__m_1/bias/m
*:(>>2Adam/m_1__to__m_2/kernel/m
$:">2Adam/m_1__to__m_2/bias/m
*:(>>2Adam/m_2__to__m_3/kernel/m
$:">2Adam/m_2__to__m_3/bias/m
*:(>>2Adam/m_3__to__m_4/kernel/m
$:">2Adam/m_3__to__m_4/bias/m
*:(>>2Adam/m_4__to__m_5/kernel/m
$:">2Adam/m_4__to__m_5/bias/m
*:(>>2Adam/m_5__to__m_6/kernel/m
$:">2Adam/m_5__to__m_6/bias/m
*:(>>2Adam/m_6__to__m_7/kernel/m
$:">2Adam/m_6__to__m_7/bias/m
*:(>>2Adam/m_7__to__m_8/kernel/m
$:">2Adam/m_7__to__m_8/bias/m
*:(>>2Adam/m_8__to__m_9/kernel/m
$:">2Adam/m_8__to__m_9/bias/m
+:)>>2Adam/m_9__to__m_10/kernel/m
%:#>2Adam/m_9__to__m_10/bias/m
,:*>>2Adam/m_10__to__m_11/kernel/m
&:$>2Adam/m_10__to__m_11/bias/m
,:*>>2Adam/m_11__to__m_12/kernel/m
&:$>2Adam/m_11__to__m_12/bias/m
,:*>>2Adam/m_12__to__m_13/kernel/m
&:$>2Adam/m_12__to__m_13/bias/m
,:*>>2Adam/m_13__to__m_14/kernel/m
&:$>2Adam/m_13__to__m_14/bias/m
,:*>>2Adam/m_14__to__m_15/kernel/m
&:$>2Adam/m_14__to__m_15/bias/m
,:*>2Adam/m_15__to__m_16/kernel/m
&:$2Adam/m_15__to__m_16/bias/m
*:(>2Adam/m_0__to__m_1/kernel/v
$:">2Adam/m_0__to__m_1/bias/v
*:(>>2Adam/m_1__to__m_2/kernel/v
$:">2Adam/m_1__to__m_2/bias/v
*:(>>2Adam/m_2__to__m_3/kernel/v
$:">2Adam/m_2__to__m_3/bias/v
*:(>>2Adam/m_3__to__m_4/kernel/v
$:">2Adam/m_3__to__m_4/bias/v
*:(>>2Adam/m_4__to__m_5/kernel/v
$:">2Adam/m_4__to__m_5/bias/v
*:(>>2Adam/m_5__to__m_6/kernel/v
$:">2Adam/m_5__to__m_6/bias/v
*:(>>2Adam/m_6__to__m_7/kernel/v
$:">2Adam/m_6__to__m_7/bias/v
*:(>>2Adam/m_7__to__m_8/kernel/v
$:">2Adam/m_7__to__m_8/bias/v
*:(>>2Adam/m_8__to__m_9/kernel/v
$:">2Adam/m_8__to__m_9/bias/v
+:)>>2Adam/m_9__to__m_10/kernel/v
%:#>2Adam/m_9__to__m_10/bias/v
,:*>>2Adam/m_10__to__m_11/kernel/v
&:$>2Adam/m_10__to__m_11/bias/v
,:*>>2Adam/m_11__to__m_12/kernel/v
&:$>2Adam/m_11__to__m_12/bias/v
,:*>>2Adam/m_12__to__m_13/kernel/v
&:$>2Adam/m_12__to__m_13/bias/v
,:*>>2Adam/m_13__to__m_14/kernel/v
&:$>2Adam/m_13__to__m_14/bias/v
,:*>>2Adam/m_14__to__m_15/kernel/v
&:$>2Adam/m_14__to__m_15/bias/v
,:*>2Adam/m_15__to__m_16/kernel/v
&:$2Adam/m_15__to__m_16/bias/v
?2?
#__inference__wrapped_model_13989509?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *1?.
,?)
m_0__to__m_1_input?????????
?2?
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13991340
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13991552
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13990775
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13990955?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_L-15-m_0-1_layer_call_fn_13990048
-__inference_L-15-m_0-1_layer_call_fn_13991621
-__inference_L-15-m_0-1_layer_call_fn_13991690
-__inference_L-15-m_0-1_layer_call_fn_13990595?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_13991713?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_m_0__to__m_1_layer_call_fn_13991722?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_13991745?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_m_1__to__m_2_layer_call_fn_13991754?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_13991777?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_m_2__to__m_3_layer_call_fn_13991786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_13991809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_m_3__to__m_4_layer_call_fn_13991818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_13991841?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_m_4__to__m_5_layer_call_fn_13991850?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_13991873?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_m_5__to__m_6_layer_call_fn_13991882?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_13991905?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_m_6__to__m_7_layer_call_fn_13991914?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_13991937?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_m_7__to__m_8_layer_call_fn_13991946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_13991969?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
/__inference_m_8__to__m_9_layer_call_fn_13991978?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_13992001?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_m_9__to__m_10_layer_call_fn_13992010?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_13992033?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_m_10__to__m_11_layer_call_fn_13992042?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_13992065?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_m_11__to__m_12_layer_call_fn_13992074?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_13992097?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_m_12__to__m_13_layer_call_fn_13992106?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_13992129?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_m_13__to__m_14_layer_call_fn_13992138?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_13992161?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_m_14__to__m_15_layer_call_fn_13992170?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_m_15__to__m_16_layer_call_and_return_conditional_losses_13992193?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_m_15__to__m_16_layer_call_fn_13992202?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
__inference_loss_fn_0_13992213?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_1_13992224?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_2_13992235?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_3_13992246?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_4_13992257?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_5_13992268?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_6_13992279?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_7_13992290?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_8_13992301?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_9_13992312?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_10_13992323?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_11_13992334?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_12_13992345?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_13_13992356?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_14_13992367?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?2?
__inference_loss_fn_15_13992378?
???
FullArgSpec
args? 
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *? 
?B?
&__inference_signature_wrapper_13991128m_0__to__m_1_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13990775? #$)*/056;<ABGHMNSTYZ_`efklqrC?@
9?6
,?)
m_0__to__m_1_input?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13990955? #$)*/056;<ABGHMNSTYZ_`efklqrC?@
9?6
,?)
m_0__to__m_1_input?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13991340? #$)*/056;<ABGHMNSTYZ_`efklqr7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_L-15-m_0-1_layer_call_and_return_conditional_losses_13991552? #$)*/056;<ABGHMNSTYZ_`efklqr7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
-__inference_L-15-m_0-1_layer_call_fn_13990048? #$)*/056;<ABGHMNSTYZ_`efklqrC?@
9?6
,?)
m_0__to__m_1_input?????????
p 

 
? "???????????
-__inference_L-15-m_0-1_layer_call_fn_13990595? #$)*/056;<ABGHMNSTYZ_`efklqrC?@
9?6
,?)
m_0__to__m_1_input?????????
p

 
? "???????????
-__inference_L-15-m_0-1_layer_call_fn_13991621u #$)*/056;<ABGHMNSTYZ_`efklqr7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
-__inference_L-15-m_0-1_layer_call_fn_13991690u #$)*/056;<ABGHMNSTYZ_`efklqr7?4
-?*
 ?
inputs?????????
p

 
? "???????????
#__inference__wrapped_model_13989509? #$)*/056;<ABGHMNSTYZ_`efklqr;?8
1?.
,?)
m_0__to__m_1_input?????????
? "??<
:
m_15__to__m_16(?%
m_15__to__m_16?????????=
__inference_loss_fn_0_13992213?

? 
? "? >
__inference_loss_fn_10_13992323S?

? 
? "? >
__inference_loss_fn_11_13992334Y?

? 
? "? >
__inference_loss_fn_12_13992345_?

? 
? "? >
__inference_loss_fn_13_13992356e?

? 
? "? >
__inference_loss_fn_14_13992367k?

? 
? "? >
__inference_loss_fn_15_13992378q?

? 
? "? =
__inference_loss_fn_1_13992224?

? 
? "? =
__inference_loss_fn_2_13992235#?

? 
? "? =
__inference_loss_fn_3_13992246)?

? 
? "? =
__inference_loss_fn_4_13992257/?

? 
? "? =
__inference_loss_fn_5_139922685?

? 
? "? =
__inference_loss_fn_6_13992279;?

? 
? "? =
__inference_loss_fn_7_13992290A?

? 
? "? =
__inference_loss_fn_8_13992301G?

? 
? "? =
__inference_loss_fn_9_13992312M?

? 
? "? ?
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_13991713\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????>
? ?
/__inference_m_0__to__m_1_layer_call_fn_13991722O/?,
%?"
 ?
inputs?????????
? "??????????>?
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_13992033\ST/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
1__inference_m_10__to__m_11_layer_call_fn_13992042OST/?,
%?"
 ?
inputs?????????>
? "??????????>?
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_13992065\YZ/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
1__inference_m_11__to__m_12_layer_call_fn_13992074OYZ/?,
%?"
 ?
inputs?????????>
? "??????????>?
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_13992097\_`/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
1__inference_m_12__to__m_13_layer_call_fn_13992106O_`/?,
%?"
 ?
inputs?????????>
? "??????????>?
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_13992129\ef/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
1__inference_m_13__to__m_14_layer_call_fn_13992138Oef/?,
%?"
 ?
inputs?????????>
? "??????????>?
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_13992161\kl/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
1__inference_m_14__to__m_15_layer_call_fn_13992170Okl/?,
%?"
 ?
inputs?????????>
? "??????????>?
L__inference_m_15__to__m_16_layer_call_and_return_conditional_losses_13992193\qr/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????
? ?
1__inference_m_15__to__m_16_layer_call_fn_13992202Oqr/?,
%?"
 ?
inputs?????????>
? "???????????
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_13991745\/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
/__inference_m_1__to__m_2_layer_call_fn_13991754O/?,
%?"
 ?
inputs?????????>
? "??????????>?
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_13991777\#$/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
/__inference_m_2__to__m_3_layer_call_fn_13991786O#$/?,
%?"
 ?
inputs?????????>
? "??????????>?
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_13991809\)*/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
/__inference_m_3__to__m_4_layer_call_fn_13991818O)*/?,
%?"
 ?
inputs?????????>
? "??????????>?
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_13991841\/0/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
/__inference_m_4__to__m_5_layer_call_fn_13991850O/0/?,
%?"
 ?
inputs?????????>
? "??????????>?
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_13991873\56/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
/__inference_m_5__to__m_6_layer_call_fn_13991882O56/?,
%?"
 ?
inputs?????????>
? "??????????>?
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_13991905\;</?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
/__inference_m_6__to__m_7_layer_call_fn_13991914O;</?,
%?"
 ?
inputs?????????>
? "??????????>?
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_13991937\AB/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
/__inference_m_7__to__m_8_layer_call_fn_13991946OAB/?,
%?"
 ?
inputs?????????>
? "??????????>?
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_13991969\GH/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
/__inference_m_8__to__m_9_layer_call_fn_13991978OGH/?,
%?"
 ?
inputs?????????>
? "??????????>?
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_13992001\MN/?,
%?"
 ?
inputs?????????>
? "%?"
?
0?????????>
? ?
0__inference_m_9__to__m_10_layer_call_fn_13992010OMN/?,
%?"
 ?
inputs?????????>
? "??????????>?
&__inference_signature_wrapper_13991128? #$)*/056;<ABGHMNSTYZ_`efklqrQ?N
? 
G?D
B
m_0__to__m_1_input,?)
m_0__to__m_1_input?????????"??<
:
m_15__to__m_16(?%
m_15__to__m_16?????????