??"
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
:1*$
shared_namem_0__to__m_1/kernel
{
'm_0__to__m_1/kernel/Read/ReadVariableOpReadVariableOpm_0__to__m_1/kernel*
_output_shapes

:1*
dtype0
z
m_0__to__m_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namem_0__to__m_1/bias
s
%m_0__to__m_1/bias/Read/ReadVariableOpReadVariableOpm_0__to__m_1/bias*
_output_shapes
:1*
dtype0
?
m_1__to__m_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*$
shared_namem_1__to__m_2/kernel
{
'm_1__to__m_2/kernel/Read/ReadVariableOpReadVariableOpm_1__to__m_2/kernel*
_output_shapes

:11*
dtype0
z
m_1__to__m_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namem_1__to__m_2/bias
s
%m_1__to__m_2/bias/Read/ReadVariableOpReadVariableOpm_1__to__m_2/bias*
_output_shapes
:1*
dtype0
?
m_2__to__m_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*$
shared_namem_2__to__m_3/kernel
{
'm_2__to__m_3/kernel/Read/ReadVariableOpReadVariableOpm_2__to__m_3/kernel*
_output_shapes

:11*
dtype0
z
m_2__to__m_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namem_2__to__m_3/bias
s
%m_2__to__m_3/bias/Read/ReadVariableOpReadVariableOpm_2__to__m_3/bias*
_output_shapes
:1*
dtype0
?
m_3__to__m_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*$
shared_namem_3__to__m_4/kernel
{
'm_3__to__m_4/kernel/Read/ReadVariableOpReadVariableOpm_3__to__m_4/kernel*
_output_shapes

:11*
dtype0
z
m_3__to__m_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namem_3__to__m_4/bias
s
%m_3__to__m_4/bias/Read/ReadVariableOpReadVariableOpm_3__to__m_4/bias*
_output_shapes
:1*
dtype0
?
m_4__to__m_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*$
shared_namem_4__to__m_5/kernel
{
'm_4__to__m_5/kernel/Read/ReadVariableOpReadVariableOpm_4__to__m_5/kernel*
_output_shapes

:11*
dtype0
z
m_4__to__m_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namem_4__to__m_5/bias
s
%m_4__to__m_5/bias/Read/ReadVariableOpReadVariableOpm_4__to__m_5/bias*
_output_shapes
:1*
dtype0
?
m_5__to__m_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*$
shared_namem_5__to__m_6/kernel
{
'm_5__to__m_6/kernel/Read/ReadVariableOpReadVariableOpm_5__to__m_6/kernel*
_output_shapes

:11*
dtype0
z
m_5__to__m_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namem_5__to__m_6/bias
s
%m_5__to__m_6/bias/Read/ReadVariableOpReadVariableOpm_5__to__m_6/bias*
_output_shapes
:1*
dtype0
?
m_6__to__m_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*$
shared_namem_6__to__m_7/kernel
{
'm_6__to__m_7/kernel/Read/ReadVariableOpReadVariableOpm_6__to__m_7/kernel*
_output_shapes

:11*
dtype0
z
m_6__to__m_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namem_6__to__m_7/bias
s
%m_6__to__m_7/bias/Read/ReadVariableOpReadVariableOpm_6__to__m_7/bias*
_output_shapes
:1*
dtype0
?
m_7__to__m_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*$
shared_namem_7__to__m_8/kernel
{
'm_7__to__m_8/kernel/Read/ReadVariableOpReadVariableOpm_7__to__m_8/kernel*
_output_shapes

:11*
dtype0
z
m_7__to__m_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namem_7__to__m_8/bias
s
%m_7__to__m_8/bias/Read/ReadVariableOpReadVariableOpm_7__to__m_8/bias*
_output_shapes
:1*
dtype0
?
m_8__to__m_9/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*$
shared_namem_8__to__m_9/kernel
{
'm_8__to__m_9/kernel/Read/ReadVariableOpReadVariableOpm_8__to__m_9/kernel*
_output_shapes

:11*
dtype0
z
m_8__to__m_9/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*"
shared_namem_8__to__m_9/bias
s
%m_8__to__m_9/bias/Read/ReadVariableOpReadVariableOpm_8__to__m_9/bias*
_output_shapes
:1*
dtype0
?
m_9__to__m_10/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*%
shared_namem_9__to__m_10/kernel
}
(m_9__to__m_10/kernel/Read/ReadVariableOpReadVariableOpm_9__to__m_10/kernel*
_output_shapes

:11*
dtype0
|
m_9__to__m_10/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*#
shared_namem_9__to__m_10/bias
u
&m_9__to__m_10/bias/Read/ReadVariableOpReadVariableOpm_9__to__m_10/bias*
_output_shapes
:1*
dtype0
?
m_10__to__m_11/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*&
shared_namem_10__to__m_11/kernel

)m_10__to__m_11/kernel/Read/ReadVariableOpReadVariableOpm_10__to__m_11/kernel*
_output_shapes

:11*
dtype0
~
m_10__to__m_11/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*$
shared_namem_10__to__m_11/bias
w
'm_10__to__m_11/bias/Read/ReadVariableOpReadVariableOpm_10__to__m_11/bias*
_output_shapes
:1*
dtype0
?
m_11__to__m_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*&
shared_namem_11__to__m_12/kernel

)m_11__to__m_12/kernel/Read/ReadVariableOpReadVariableOpm_11__to__m_12/kernel*
_output_shapes

:11*
dtype0
~
m_11__to__m_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*$
shared_namem_11__to__m_12/bias
w
'm_11__to__m_12/bias/Read/ReadVariableOpReadVariableOpm_11__to__m_12/bias*
_output_shapes
:1*
dtype0
?
m_12__to__m_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*&
shared_namem_12__to__m_13/kernel

)m_12__to__m_13/kernel/Read/ReadVariableOpReadVariableOpm_12__to__m_13/kernel*
_output_shapes

:11*
dtype0
~
m_12__to__m_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*$
shared_namem_12__to__m_13/bias
w
'm_12__to__m_13/bias/Read/ReadVariableOpReadVariableOpm_12__to__m_13/bias*
_output_shapes
:1*
dtype0
?
m_13__to__m_14/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*&
shared_namem_13__to__m_14/kernel

)m_13__to__m_14/kernel/Read/ReadVariableOpReadVariableOpm_13__to__m_14/kernel*
_output_shapes

:11*
dtype0
~
m_13__to__m_14/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*$
shared_namem_13__to__m_14/bias
w
'm_13__to__m_14/bias/Read/ReadVariableOpReadVariableOpm_13__to__m_14/bias*
_output_shapes
:1*
dtype0
?
m_14__to__m_15/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:1*&
shared_namem_14__to__m_15/kernel

)m_14__to__m_15/kernel/Read/ReadVariableOpReadVariableOpm_14__to__m_15/kernel*
_output_shapes

:1*
dtype0
~
m_14__to__m_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namem_14__to__m_15/bias
w
'm_14__to__m_15/bias/Read/ReadVariableOpReadVariableOpm_14__to__m_15/bias*
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
:1*+
shared_nameAdam/m_0__to__m_1/kernel/m
?
.Adam/m_0__to__m_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_0__to__m_1/kernel/m*
_output_shapes

:1*
dtype0
?
Adam/m_0__to__m_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_0__to__m_1/bias/m
?
,Adam/m_0__to__m_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_0__to__m_1/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_1__to__m_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_1__to__m_2/kernel/m
?
.Adam/m_1__to__m_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_1__to__m_2/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_1__to__m_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_1__to__m_2/bias/m
?
,Adam/m_1__to__m_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_1__to__m_2/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_2__to__m_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_2__to__m_3/kernel/m
?
.Adam/m_2__to__m_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_2__to__m_3/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_2__to__m_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_2__to__m_3/bias/m
?
,Adam/m_2__to__m_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_2__to__m_3/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_3__to__m_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_3__to__m_4/kernel/m
?
.Adam/m_3__to__m_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_3__to__m_4/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_3__to__m_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_3__to__m_4/bias/m
?
,Adam/m_3__to__m_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_3__to__m_4/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_4__to__m_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_4__to__m_5/kernel/m
?
.Adam/m_4__to__m_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_4__to__m_5/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_4__to__m_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_4__to__m_5/bias/m
?
,Adam/m_4__to__m_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_4__to__m_5/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_5__to__m_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_5__to__m_6/kernel/m
?
.Adam/m_5__to__m_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_5__to__m_6/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_5__to__m_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_5__to__m_6/bias/m
?
,Adam/m_5__to__m_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_5__to__m_6/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_6__to__m_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_6__to__m_7/kernel/m
?
.Adam/m_6__to__m_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_6__to__m_7/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_6__to__m_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_6__to__m_7/bias/m
?
,Adam/m_6__to__m_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_6__to__m_7/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_7__to__m_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_7__to__m_8/kernel/m
?
.Adam/m_7__to__m_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_7__to__m_8/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_7__to__m_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_7__to__m_8/bias/m
?
,Adam/m_7__to__m_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_7__to__m_8/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_8__to__m_9/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_8__to__m_9/kernel/m
?
.Adam/m_8__to__m_9/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_8__to__m_9/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_8__to__m_9/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_8__to__m_9/bias/m
?
,Adam/m_8__to__m_9/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_8__to__m_9/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_9__to__m_10/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*,
shared_nameAdam/m_9__to__m_10/kernel/m
?
/Adam/m_9__to__m_10/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_9__to__m_10/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_9__to__m_10/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1**
shared_nameAdam/m_9__to__m_10/bias/m
?
-Adam/m_9__to__m_10/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_9__to__m_10/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_10__to__m_11/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*-
shared_nameAdam/m_10__to__m_11/kernel/m
?
0Adam/m_10__to__m_11/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_10__to__m_11/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_10__to__m_11/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*+
shared_nameAdam/m_10__to__m_11/bias/m
?
.Adam/m_10__to__m_11/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_10__to__m_11/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_11__to__m_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*-
shared_nameAdam/m_11__to__m_12/kernel/m
?
0Adam/m_11__to__m_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_11__to__m_12/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_11__to__m_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*+
shared_nameAdam/m_11__to__m_12/bias/m
?
.Adam/m_11__to__m_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_11__to__m_12/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_12__to__m_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*-
shared_nameAdam/m_12__to__m_13/kernel/m
?
0Adam/m_12__to__m_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_12__to__m_13/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_12__to__m_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*+
shared_nameAdam/m_12__to__m_13/bias/m
?
.Adam/m_12__to__m_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_12__to__m_13/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_13__to__m_14/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*-
shared_nameAdam/m_13__to__m_14/kernel/m
?
0Adam/m_13__to__m_14/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_13__to__m_14/kernel/m*
_output_shapes

:11*
dtype0
?
Adam/m_13__to__m_14/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*+
shared_nameAdam/m_13__to__m_14/bias/m
?
.Adam/m_13__to__m_14/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_13__to__m_14/bias/m*
_output_shapes
:1*
dtype0
?
Adam/m_14__to__m_15/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:1*-
shared_nameAdam/m_14__to__m_15/kernel/m
?
0Adam/m_14__to__m_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/kernel/m*
_output_shapes

:1*
dtype0
?
Adam/m_14__to__m_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m_14__to__m_15/bias/m
?
.Adam/m_14__to__m_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/bias/m*
_output_shapes
:*
dtype0
?
Adam/m_0__to__m_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:1*+
shared_nameAdam/m_0__to__m_1/kernel/v
?
.Adam/m_0__to__m_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_0__to__m_1/kernel/v*
_output_shapes

:1*
dtype0
?
Adam/m_0__to__m_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_0__to__m_1/bias/v
?
,Adam/m_0__to__m_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_0__to__m_1/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_1__to__m_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_1__to__m_2/kernel/v
?
.Adam/m_1__to__m_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_1__to__m_2/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_1__to__m_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_1__to__m_2/bias/v
?
,Adam/m_1__to__m_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_1__to__m_2/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_2__to__m_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_2__to__m_3/kernel/v
?
.Adam/m_2__to__m_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_2__to__m_3/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_2__to__m_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_2__to__m_3/bias/v
?
,Adam/m_2__to__m_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_2__to__m_3/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_3__to__m_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_3__to__m_4/kernel/v
?
.Adam/m_3__to__m_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_3__to__m_4/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_3__to__m_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_3__to__m_4/bias/v
?
,Adam/m_3__to__m_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_3__to__m_4/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_4__to__m_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_4__to__m_5/kernel/v
?
.Adam/m_4__to__m_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_4__to__m_5/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_4__to__m_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_4__to__m_5/bias/v
?
,Adam/m_4__to__m_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_4__to__m_5/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_5__to__m_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_5__to__m_6/kernel/v
?
.Adam/m_5__to__m_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_5__to__m_6/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_5__to__m_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_5__to__m_6/bias/v
?
,Adam/m_5__to__m_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_5__to__m_6/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_6__to__m_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_6__to__m_7/kernel/v
?
.Adam/m_6__to__m_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_6__to__m_7/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_6__to__m_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_6__to__m_7/bias/v
?
,Adam/m_6__to__m_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_6__to__m_7/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_7__to__m_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_7__to__m_8/kernel/v
?
.Adam/m_7__to__m_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_7__to__m_8/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_7__to__m_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_7__to__m_8/bias/v
?
,Adam/m_7__to__m_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_7__to__m_8/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_8__to__m_9/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*+
shared_nameAdam/m_8__to__m_9/kernel/v
?
.Adam/m_8__to__m_9/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_8__to__m_9/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_8__to__m_9/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*)
shared_nameAdam/m_8__to__m_9/bias/v
?
,Adam/m_8__to__m_9/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_8__to__m_9/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_9__to__m_10/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*,
shared_nameAdam/m_9__to__m_10/kernel/v
?
/Adam/m_9__to__m_10/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_9__to__m_10/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_9__to__m_10/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1**
shared_nameAdam/m_9__to__m_10/bias/v
?
-Adam/m_9__to__m_10/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_9__to__m_10/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_10__to__m_11/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*-
shared_nameAdam/m_10__to__m_11/kernel/v
?
0Adam/m_10__to__m_11/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_10__to__m_11/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_10__to__m_11/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*+
shared_nameAdam/m_10__to__m_11/bias/v
?
.Adam/m_10__to__m_11/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_10__to__m_11/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_11__to__m_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*-
shared_nameAdam/m_11__to__m_12/kernel/v
?
0Adam/m_11__to__m_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_11__to__m_12/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_11__to__m_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*+
shared_nameAdam/m_11__to__m_12/bias/v
?
.Adam/m_11__to__m_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_11__to__m_12/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_12__to__m_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*-
shared_nameAdam/m_12__to__m_13/kernel/v
?
0Adam/m_12__to__m_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_12__to__m_13/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_12__to__m_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*+
shared_nameAdam/m_12__to__m_13/bias/v
?
.Adam/m_12__to__m_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_12__to__m_13/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_13__to__m_14/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:11*-
shared_nameAdam/m_13__to__m_14/kernel/v
?
0Adam/m_13__to__m_14/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_13__to__m_14/kernel/v*
_output_shapes

:11*
dtype0
?
Adam/m_13__to__m_14/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:1*+
shared_nameAdam/m_13__to__m_14/bias/v
?
.Adam/m_13__to__m_14/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_13__to__m_14/bias/v*
_output_shapes
:1*
dtype0
?
Adam/m_14__to__m_15/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:1*-
shared_nameAdam/m_14__to__m_15/kernel/v
?
0Adam/m_14__to__m_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/kernel/v*
_output_shapes

:1*
dtype0
?
Adam/m_14__to__m_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m_14__to__m_15/bias/v
?
.Adam/m_14__to__m_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
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
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
h

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
h

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
h

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
h

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
h

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
h

@kernel
Abias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
h

Fkernel
Gbias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
h

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
h

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
h

Xkernel
Ybias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
h

^kernel
_bias
`regularization_losses
a	variables
btrainable_variables
c	keras_api
h

dkernel
ebias
fregularization_losses
g	variables
htrainable_variables
i	keras_api
h

jkernel
kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
?
piter

qbeta_1

rbeta_2
	sdecay
tlearning_ratem?m?m?m?"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?@m?Am?Fm?Gm?Lm?Mm?Rm?Sm?Xm?Ym?^m?_m?dm?em?jm?km?v?v?v?v?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?@v?Av?Fv?Gv?Lv?Mv?Rv?Sv?Xv?Yv?^v?_v?dv?ev?jv?kv?
 
?
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511
:12
;13
@14
A15
F16
G17
L18
M19
R20
S21
X22
Y23
^24
_25
d26
e27
j28
k29
?
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511
:12
;13
@14
A15
F16
G17
L18
M19
R20
S21
X22
Y23
^24
_25
d26
e27
j28
k29
?
regularization_losses
ulayer_regularization_losses
vmetrics
wnon_trainable_variables

xlayers
	variables
trainable_variables
ylayer_metrics
 
_]
VARIABLE_VALUEm_0__to__m_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_0__to__m_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
zlayer_regularization_losses
{metrics
|non_trainable_variables

}layers
	variables
trainable_variables
~layer_metrics
_]
VARIABLE_VALUEm_1__to__m_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_1__to__m_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
regularization_losses
layer_regularization_losses
?metrics
?non_trainable_variables
?layers
	variables
 trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_2__to__m_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_2__to__m_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

"0
#1

"0
#1
?
$regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
%	variables
&trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_3__to__m_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_3__to__m_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

(0
)1

(0
)1
?
*regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
+	variables
,trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_4__to__m_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_4__to__m_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

.0
/1

.0
/1
?
0regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
1	variables
2trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_5__to__m_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_5__to__m_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

40
51

40
51
?
6regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
7	variables
8trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_6__to__m_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_6__to__m_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

:0
;1

:0
;1
?
<regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
=	variables
>trainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_7__to__m_8/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_7__to__m_8/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

@0
A1

@0
A1
?
Bregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
C	variables
Dtrainable_variables
?layer_metrics
_]
VARIABLE_VALUEm_8__to__m_9/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_8__to__m_9/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

F0
G1

F0
G1
?
Hregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
I	variables
Jtrainable_variables
?layer_metrics
`^
VARIABLE_VALUEm_9__to__m_10/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEm_9__to__m_10/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

L0
M1

L0
M1
?
Nregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
O	variables
Ptrainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_10__to__m_11/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_10__to__m_11/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

R0
S1

R0
S1
?
Tregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
U	variables
Vtrainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_11__to__m_12/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_11__to__m_12/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE
 

X0
Y1

X0
Y1
?
Zregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
[	variables
\trainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_12__to__m_13/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_12__to__m_13/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

^0
_1

^0
_1
?
`regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
a	variables
btrainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_13__to__m_14/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_13__to__m_14/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE
 

d0
e1

d0
e1
?
fregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
g	variables
htrainable_variables
?layer_metrics
b`
VARIABLE_VALUEm_14__to__m_15/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_14__to__m_15/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE
 

j0
k1

j0
k1
?
lregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
m	variables
ntrainable_variables
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
n
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
?
"serving_default_m_0__to__m_1_inputPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_m_0__to__m_1_inputm_0__to__m_1/kernelm_0__to__m_1/biasm_1__to__m_2/kernelm_1__to__m_2/biasm_2__to__m_3/kernelm_2__to__m_3/biasm_3__to__m_4/kernelm_3__to__m_4/biasm_4__to__m_5/kernelm_4__to__m_5/biasm_5__to__m_6/kernelm_5__to__m_6/biasm_6__to__m_7/kernelm_6__to__m_7/biasm_7__to__m_8/kernelm_7__to__m_8/biasm_8__to__m_9/kernelm_8__to__m_9/biasm_9__to__m_10/kernelm_9__to__m_10/biasm_10__to__m_11/kernelm_10__to__m_11/biasm_11__to__m_12/kernelm_11__to__m_12/biasm_12__to__m_13/kernelm_12__to__m_13/biasm_13__to__m_14/kernelm_13__to__m_14/biasm_14__to__m_15/kernelm_14__to__m_15/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_23585200
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?$
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename'm_0__to__m_1/kernel/Read/ReadVariableOp%m_0__to__m_1/bias/Read/ReadVariableOp'm_1__to__m_2/kernel/Read/ReadVariableOp%m_1__to__m_2/bias/Read/ReadVariableOp'm_2__to__m_3/kernel/Read/ReadVariableOp%m_2__to__m_3/bias/Read/ReadVariableOp'm_3__to__m_4/kernel/Read/ReadVariableOp%m_3__to__m_4/bias/Read/ReadVariableOp'm_4__to__m_5/kernel/Read/ReadVariableOp%m_4__to__m_5/bias/Read/ReadVariableOp'm_5__to__m_6/kernel/Read/ReadVariableOp%m_5__to__m_6/bias/Read/ReadVariableOp'm_6__to__m_7/kernel/Read/ReadVariableOp%m_6__to__m_7/bias/Read/ReadVariableOp'm_7__to__m_8/kernel/Read/ReadVariableOp%m_7__to__m_8/bias/Read/ReadVariableOp'm_8__to__m_9/kernel/Read/ReadVariableOp%m_8__to__m_9/bias/Read/ReadVariableOp(m_9__to__m_10/kernel/Read/ReadVariableOp&m_9__to__m_10/bias/Read/ReadVariableOp)m_10__to__m_11/kernel/Read/ReadVariableOp'm_10__to__m_11/bias/Read/ReadVariableOp)m_11__to__m_12/kernel/Read/ReadVariableOp'm_11__to__m_12/bias/Read/ReadVariableOp)m_12__to__m_13/kernel/Read/ReadVariableOp'm_12__to__m_13/bias/Read/ReadVariableOp)m_13__to__m_14/kernel/Read/ReadVariableOp'm_13__to__m_14/bias/Read/ReadVariableOp)m_14__to__m_15/kernel/Read/ReadVariableOp'm_14__to__m_15/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp.Adam/m_0__to__m_1/kernel/m/Read/ReadVariableOp,Adam/m_0__to__m_1/bias/m/Read/ReadVariableOp.Adam/m_1__to__m_2/kernel/m/Read/ReadVariableOp,Adam/m_1__to__m_2/bias/m/Read/ReadVariableOp.Adam/m_2__to__m_3/kernel/m/Read/ReadVariableOp,Adam/m_2__to__m_3/bias/m/Read/ReadVariableOp.Adam/m_3__to__m_4/kernel/m/Read/ReadVariableOp,Adam/m_3__to__m_4/bias/m/Read/ReadVariableOp.Adam/m_4__to__m_5/kernel/m/Read/ReadVariableOp,Adam/m_4__to__m_5/bias/m/Read/ReadVariableOp.Adam/m_5__to__m_6/kernel/m/Read/ReadVariableOp,Adam/m_5__to__m_6/bias/m/Read/ReadVariableOp.Adam/m_6__to__m_7/kernel/m/Read/ReadVariableOp,Adam/m_6__to__m_7/bias/m/Read/ReadVariableOp.Adam/m_7__to__m_8/kernel/m/Read/ReadVariableOp,Adam/m_7__to__m_8/bias/m/Read/ReadVariableOp.Adam/m_8__to__m_9/kernel/m/Read/ReadVariableOp,Adam/m_8__to__m_9/bias/m/Read/ReadVariableOp/Adam/m_9__to__m_10/kernel/m/Read/ReadVariableOp-Adam/m_9__to__m_10/bias/m/Read/ReadVariableOp0Adam/m_10__to__m_11/kernel/m/Read/ReadVariableOp.Adam/m_10__to__m_11/bias/m/Read/ReadVariableOp0Adam/m_11__to__m_12/kernel/m/Read/ReadVariableOp.Adam/m_11__to__m_12/bias/m/Read/ReadVariableOp0Adam/m_12__to__m_13/kernel/m/Read/ReadVariableOp.Adam/m_12__to__m_13/bias/m/Read/ReadVariableOp0Adam/m_13__to__m_14/kernel/m/Read/ReadVariableOp.Adam/m_13__to__m_14/bias/m/Read/ReadVariableOp0Adam/m_14__to__m_15/kernel/m/Read/ReadVariableOp.Adam/m_14__to__m_15/bias/m/Read/ReadVariableOp.Adam/m_0__to__m_1/kernel/v/Read/ReadVariableOp,Adam/m_0__to__m_1/bias/v/Read/ReadVariableOp.Adam/m_1__to__m_2/kernel/v/Read/ReadVariableOp,Adam/m_1__to__m_2/bias/v/Read/ReadVariableOp.Adam/m_2__to__m_3/kernel/v/Read/ReadVariableOp,Adam/m_2__to__m_3/bias/v/Read/ReadVariableOp.Adam/m_3__to__m_4/kernel/v/Read/ReadVariableOp,Adam/m_3__to__m_4/bias/v/Read/ReadVariableOp.Adam/m_4__to__m_5/kernel/v/Read/ReadVariableOp,Adam/m_4__to__m_5/bias/v/Read/ReadVariableOp.Adam/m_5__to__m_6/kernel/v/Read/ReadVariableOp,Adam/m_5__to__m_6/bias/v/Read/ReadVariableOp.Adam/m_6__to__m_7/kernel/v/Read/ReadVariableOp,Adam/m_6__to__m_7/bias/v/Read/ReadVariableOp.Adam/m_7__to__m_8/kernel/v/Read/ReadVariableOp,Adam/m_7__to__m_8/bias/v/Read/ReadVariableOp.Adam/m_8__to__m_9/kernel/v/Read/ReadVariableOp,Adam/m_8__to__m_9/bias/v/Read/ReadVariableOp/Adam/m_9__to__m_10/kernel/v/Read/ReadVariableOp-Adam/m_9__to__m_10/bias/v/Read/ReadVariableOp0Adam/m_10__to__m_11/kernel/v/Read/ReadVariableOp.Adam/m_10__to__m_11/bias/v/Read/ReadVariableOp0Adam/m_11__to__m_12/kernel/v/Read/ReadVariableOp.Adam/m_11__to__m_12/bias/v/Read/ReadVariableOp0Adam/m_12__to__m_13/kernel/v/Read/ReadVariableOp.Adam/m_12__to__m_13/bias/v/Read/ReadVariableOp0Adam/m_13__to__m_14/kernel/v/Read/ReadVariableOp.Adam/m_13__to__m_14/bias/v/Read/ReadVariableOp0Adam/m_14__to__m_15/kernel/v/Read/ReadVariableOp.Adam/m_14__to__m_15/bias/v/Read/ReadVariableOpConst*n
Ting
e2c	*
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
!__inference__traced_save_23586687
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamem_0__to__m_1/kernelm_0__to__m_1/biasm_1__to__m_2/kernelm_1__to__m_2/biasm_2__to__m_3/kernelm_2__to__m_3/biasm_3__to__m_4/kernelm_3__to__m_4/biasm_4__to__m_5/kernelm_4__to__m_5/biasm_5__to__m_6/kernelm_5__to__m_6/biasm_6__to__m_7/kernelm_6__to__m_7/biasm_7__to__m_8/kernelm_7__to__m_8/biasm_8__to__m_9/kernelm_8__to__m_9/biasm_9__to__m_10/kernelm_9__to__m_10/biasm_10__to__m_11/kernelm_10__to__m_11/biasm_11__to__m_12/kernelm_11__to__m_12/biasm_12__to__m_13/kernelm_12__to__m_13/biasm_13__to__m_14/kernelm_13__to__m_14/biasm_14__to__m_15/kernelm_14__to__m_15/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/m_0__to__m_1/kernel/mAdam/m_0__to__m_1/bias/mAdam/m_1__to__m_2/kernel/mAdam/m_1__to__m_2/bias/mAdam/m_2__to__m_3/kernel/mAdam/m_2__to__m_3/bias/mAdam/m_3__to__m_4/kernel/mAdam/m_3__to__m_4/bias/mAdam/m_4__to__m_5/kernel/mAdam/m_4__to__m_5/bias/mAdam/m_5__to__m_6/kernel/mAdam/m_5__to__m_6/bias/mAdam/m_6__to__m_7/kernel/mAdam/m_6__to__m_7/bias/mAdam/m_7__to__m_8/kernel/mAdam/m_7__to__m_8/bias/mAdam/m_8__to__m_9/kernel/mAdam/m_8__to__m_9/bias/mAdam/m_9__to__m_10/kernel/mAdam/m_9__to__m_10/bias/mAdam/m_10__to__m_11/kernel/mAdam/m_10__to__m_11/bias/mAdam/m_11__to__m_12/kernel/mAdam/m_11__to__m_12/bias/mAdam/m_12__to__m_13/kernel/mAdam/m_12__to__m_13/bias/mAdam/m_13__to__m_14/kernel/mAdam/m_13__to__m_14/bias/mAdam/m_14__to__m_15/kernel/mAdam/m_14__to__m_15/bias/mAdam/m_0__to__m_1/kernel/vAdam/m_0__to__m_1/bias/vAdam/m_1__to__m_2/kernel/vAdam/m_1__to__m_2/bias/vAdam/m_2__to__m_3/kernel/vAdam/m_2__to__m_3/bias/vAdam/m_3__to__m_4/kernel/vAdam/m_3__to__m_4/bias/vAdam/m_4__to__m_5/kernel/vAdam/m_4__to__m_5/bias/vAdam/m_5__to__m_6/kernel/vAdam/m_5__to__m_6/bias/vAdam/m_6__to__m_7/kernel/vAdam/m_6__to__m_7/bias/vAdam/m_7__to__m_8/kernel/vAdam/m_7__to__m_8/bias/vAdam/m_8__to__m_9/kernel/vAdam/m_8__to__m_9/bias/vAdam/m_9__to__m_10/kernel/vAdam/m_9__to__m_10/bias/vAdam/m_10__to__m_11/kernel/vAdam/m_10__to__m_11/bias/vAdam/m_11__to__m_12/kernel/vAdam/m_11__to__m_12/bias/vAdam/m_12__to__m_13/kernel/vAdam/m_12__to__m_13/bias/vAdam/m_13__to__m_14/kernel/vAdam/m_13__to__m_14/bias/vAdam/m_14__to__m_15/kernel/vAdam/m_14__to__m_15/bias/v*m
Tinf
d2b*
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
$__inference__traced_restore_23586988??
?
?
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_23585815

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_23583749

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
??
?
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23584122

inputs'
m_0__to__m_1_23583704:1#
m_0__to__m_1_23583706:1'
m_1__to__m_2_23583727:11#
m_1__to__m_2_23583729:1'
m_2__to__m_3_23583750:11#
m_2__to__m_3_23583752:1'
m_3__to__m_4_23583773:11#
m_3__to__m_4_23583775:1'
m_4__to__m_5_23583796:11#
m_4__to__m_5_23583798:1'
m_5__to__m_6_23583819:11#
m_5__to__m_6_23583821:1'
m_6__to__m_7_23583842:11#
m_6__to__m_7_23583844:1'
m_7__to__m_8_23583865:11#
m_7__to__m_8_23583867:1'
m_8__to__m_9_23583888:11#
m_8__to__m_9_23583890:1(
m_9__to__m_10_23583911:11$
m_9__to__m_10_23583913:1)
m_10__to__m_11_23583934:11%
m_10__to__m_11_23583936:1)
m_11__to__m_12_23583957:11%
m_11__to__m_12_23583959:1)
m_12__to__m_13_23583980:11%
m_12__to__m_13_23583982:1)
m_13__to__m_14_23584003:11%
m_13__to__m_14_23584005:1)
m_14__to__m_15_23584026:1%
m_14__to__m_15_23584028:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallinputsm_0__to__m_1_23583704m_0__to__m_1_23583706*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_235837032&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_23583727m_1__to__m_2_23583729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_235837262&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_23583750m_2__to__m_3_23583752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_235837492&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_23583773m_3__to__m_4_23583775*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_235837722&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_23583796m_4__to__m_5_23583798*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_235837952&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_23583819m_5__to__m_6_23583821*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_235838182&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_23583842m_6__to__m_7_23583844*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_235838412&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_23583865m_7__to__m_8_23583867*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_235838642&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_23583888m_8__to__m_9_23583890*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_235838872&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_23583911m_9__to__m_10_23583913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_235839102'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_23583934m_10__to__m_11_23583936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_235839332(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_23583957m_11__to__m_12_23583959*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_235839562(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_23583980m_12__to__m_13_23583982*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_235839792(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_23584003m_13__to__m_14_23584005*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_235840022(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_23584026m_14__to__m_15_23584028*
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
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_235840252(
&m_14__to__m_15/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_23583704*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_23583727*
_output_shapes

:11*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_23583750*
_output_shapes

:11*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_23583773*
_output_shapes

:11*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_23583796*
_output_shapes

:11*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_23583819*
_output_shapes

:11*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_23583842*
_output_shapes

:11*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_23583865*
_output_shapes

:11*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_23583888*
_output_shapes

:11*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_23583911*
_output_shapes

:11*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112&
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
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_23583934*
_output_shapes

:11*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_23583957*
_output_shapes

:11*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_23583980*
_output_shapes

:11*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_23584003*
_output_shapes

:11*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_23584026*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
%m_14__to__m_15/kernel/Regularizer/mul?
IdentityIdentity/m_14__to__m_15/StatefulPartitionedCall:output:0%^m_0__to__m_1/StatefulPartitionedCall3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp'^m_10__to__m_11/StatefulPartitionedCall5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp'^m_11__to__m_12/StatefulPartitionedCall5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp'^m_12__to__m_13/StatefulPartitionedCall5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp'^m_13__to__m_14/StatefulPartitionedCall5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp'^m_14__to__m_15/StatefulPartitionedCall5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp%^m_1__to__m_2/StatefulPartitionedCall3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp%^m_2__to__m_3/StatefulPartitionedCall3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp%^m_3__to__m_4/StatefulPartitionedCall3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp%^m_4__to__m_5/StatefulPartitionedCall3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp%^m_5__to__m_6/StatefulPartitionedCall3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp%^m_6__to__m_7/StatefulPartitionedCall3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp%^m_7__to__m_8/StatefulPartitionedCall3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp%^m_8__to__m_9/StatefulPartitionedCall3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp&^m_9__to__m_10/StatefulPartitionedCall4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2L
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
__inference_loss_fn_0_23586219M
;m_0__to__m_1_kernel_regularizer_abs_readvariableop_resource:1
identity??2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_0__to__m_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
?
?
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_23586135

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_23586167

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
??
?*
!__inference__traced_save_23586687
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
.savev2_m_14__to__m_15_bias_read_readvariableop(
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
5savev2_adam_m_14__to__m_15_bias_m_read_readvariableop9
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
5savev2_adam_m_14__to__m_15_bias_v_read_readvariableop
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
ShardedFilename?7
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?7
value?6B?6bB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?
value?B?bB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?)
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0.savev2_m_0__to__m_1_kernel_read_readvariableop,savev2_m_0__to__m_1_bias_read_readvariableop.savev2_m_1__to__m_2_kernel_read_readvariableop,savev2_m_1__to__m_2_bias_read_readvariableop.savev2_m_2__to__m_3_kernel_read_readvariableop,savev2_m_2__to__m_3_bias_read_readvariableop.savev2_m_3__to__m_4_kernel_read_readvariableop,savev2_m_3__to__m_4_bias_read_readvariableop.savev2_m_4__to__m_5_kernel_read_readvariableop,savev2_m_4__to__m_5_bias_read_readvariableop.savev2_m_5__to__m_6_kernel_read_readvariableop,savev2_m_5__to__m_6_bias_read_readvariableop.savev2_m_6__to__m_7_kernel_read_readvariableop,savev2_m_6__to__m_7_bias_read_readvariableop.savev2_m_7__to__m_8_kernel_read_readvariableop,savev2_m_7__to__m_8_bias_read_readvariableop.savev2_m_8__to__m_9_kernel_read_readvariableop,savev2_m_8__to__m_9_bias_read_readvariableop/savev2_m_9__to__m_10_kernel_read_readvariableop-savev2_m_9__to__m_10_bias_read_readvariableop0savev2_m_10__to__m_11_kernel_read_readvariableop.savev2_m_10__to__m_11_bias_read_readvariableop0savev2_m_11__to__m_12_kernel_read_readvariableop.savev2_m_11__to__m_12_bias_read_readvariableop0savev2_m_12__to__m_13_kernel_read_readvariableop.savev2_m_12__to__m_13_bias_read_readvariableop0savev2_m_13__to__m_14_kernel_read_readvariableop.savev2_m_13__to__m_14_bias_read_readvariableop0savev2_m_14__to__m_15_kernel_read_readvariableop.savev2_m_14__to__m_15_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop5savev2_adam_m_0__to__m_1_kernel_m_read_readvariableop3savev2_adam_m_0__to__m_1_bias_m_read_readvariableop5savev2_adam_m_1__to__m_2_kernel_m_read_readvariableop3savev2_adam_m_1__to__m_2_bias_m_read_readvariableop5savev2_adam_m_2__to__m_3_kernel_m_read_readvariableop3savev2_adam_m_2__to__m_3_bias_m_read_readvariableop5savev2_adam_m_3__to__m_4_kernel_m_read_readvariableop3savev2_adam_m_3__to__m_4_bias_m_read_readvariableop5savev2_adam_m_4__to__m_5_kernel_m_read_readvariableop3savev2_adam_m_4__to__m_5_bias_m_read_readvariableop5savev2_adam_m_5__to__m_6_kernel_m_read_readvariableop3savev2_adam_m_5__to__m_6_bias_m_read_readvariableop5savev2_adam_m_6__to__m_7_kernel_m_read_readvariableop3savev2_adam_m_6__to__m_7_bias_m_read_readvariableop5savev2_adam_m_7__to__m_8_kernel_m_read_readvariableop3savev2_adam_m_7__to__m_8_bias_m_read_readvariableop5savev2_adam_m_8__to__m_9_kernel_m_read_readvariableop3savev2_adam_m_8__to__m_9_bias_m_read_readvariableop6savev2_adam_m_9__to__m_10_kernel_m_read_readvariableop4savev2_adam_m_9__to__m_10_bias_m_read_readvariableop7savev2_adam_m_10__to__m_11_kernel_m_read_readvariableop5savev2_adam_m_10__to__m_11_bias_m_read_readvariableop7savev2_adam_m_11__to__m_12_kernel_m_read_readvariableop5savev2_adam_m_11__to__m_12_bias_m_read_readvariableop7savev2_adam_m_12__to__m_13_kernel_m_read_readvariableop5savev2_adam_m_12__to__m_13_bias_m_read_readvariableop7savev2_adam_m_13__to__m_14_kernel_m_read_readvariableop5savev2_adam_m_13__to__m_14_bias_m_read_readvariableop7savev2_adam_m_14__to__m_15_kernel_m_read_readvariableop5savev2_adam_m_14__to__m_15_bias_m_read_readvariableop5savev2_adam_m_0__to__m_1_kernel_v_read_readvariableop3savev2_adam_m_0__to__m_1_bias_v_read_readvariableop5savev2_adam_m_1__to__m_2_kernel_v_read_readvariableop3savev2_adam_m_1__to__m_2_bias_v_read_readvariableop5savev2_adam_m_2__to__m_3_kernel_v_read_readvariableop3savev2_adam_m_2__to__m_3_bias_v_read_readvariableop5savev2_adam_m_3__to__m_4_kernel_v_read_readvariableop3savev2_adam_m_3__to__m_4_bias_v_read_readvariableop5savev2_adam_m_4__to__m_5_kernel_v_read_readvariableop3savev2_adam_m_4__to__m_5_bias_v_read_readvariableop5savev2_adam_m_5__to__m_6_kernel_v_read_readvariableop3savev2_adam_m_5__to__m_6_bias_v_read_readvariableop5savev2_adam_m_6__to__m_7_kernel_v_read_readvariableop3savev2_adam_m_6__to__m_7_bias_v_read_readvariableop5savev2_adam_m_7__to__m_8_kernel_v_read_readvariableop3savev2_adam_m_7__to__m_8_bias_v_read_readvariableop5savev2_adam_m_8__to__m_9_kernel_v_read_readvariableop3savev2_adam_m_8__to__m_9_bias_v_read_readvariableop6savev2_adam_m_9__to__m_10_kernel_v_read_readvariableop4savev2_adam_m_9__to__m_10_bias_v_read_readvariableop7savev2_adam_m_10__to__m_11_kernel_v_read_readvariableop5savev2_adam_m_10__to__m_11_bias_v_read_readvariableop7savev2_adam_m_11__to__m_12_kernel_v_read_readvariableop5savev2_adam_m_11__to__m_12_bias_v_read_readvariableop7savev2_adam_m_12__to__m_13_kernel_v_read_readvariableop5savev2_adam_m_12__to__m_13_bias_v_read_readvariableop7savev2_adam_m_13__to__m_14_kernel_v_read_readvariableop5savev2_adam_m_13__to__m_14_bias_v_read_readvariableop7savev2_adam_m_14__to__m_15_kernel_v_read_readvariableop5savev2_adam_m_14__to__m_15_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *p
dtypesf
d2b	2
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

identity_1Identity_1:output:0*?
_input_shapes?
?: :1:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:1:: : : : : : : :1:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:1::1:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:1:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:1: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$	 

_output_shapes

:11: 


_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:11: 

_output_shapes
:1:$ 

_output_shapes

:1: 

_output_shapes
::

_output_shapes
: : 

_output_shapes
: :!
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
: :$& 

_output_shapes

:1: '

_output_shapes
:1:$( 

_output_shapes

:11: )

_output_shapes
:1:$* 

_output_shapes

:11: +

_output_shapes
:1:$, 

_output_shapes

:11: -

_output_shapes
:1:$. 

_output_shapes

:11: /

_output_shapes
:1:$0 

_output_shapes

:11: 1

_output_shapes
:1:$2 

_output_shapes

:11: 3

_output_shapes
:1:$4 

_output_shapes

:11: 5

_output_shapes
:1:$6 

_output_shapes

:11: 7

_output_shapes
:1:$8 

_output_shapes

:11: 9

_output_shapes
:1:$: 

_output_shapes

:11: ;

_output_shapes
:1:$< 

_output_shapes

:11: =

_output_shapes
:1:$> 

_output_shapes

:11: ?

_output_shapes
:1:$@ 

_output_shapes

:11: A

_output_shapes
:1:$B 

_output_shapes

:1: C

_output_shapes
::$D 

_output_shapes

:1: E

_output_shapes
:1:$F 

_output_shapes

:11: G

_output_shapes
:1:$H 

_output_shapes

:11: I

_output_shapes
:1:$J 

_output_shapes

:11: K

_output_shapes
:1:$L 

_output_shapes

:11: M

_output_shapes
:1:$N 

_output_shapes

:11: O

_output_shapes
:1:$P 

_output_shapes

:11: Q

_output_shapes
:1:$R 

_output_shapes

:11: S

_output_shapes
:1:$T 

_output_shapes

:11: U

_output_shapes
:1:$V 

_output_shapes

:11: W

_output_shapes
:1:$X 

_output_shapes

:11: Y

_output_shapes
:1:$Z 

_output_shapes

:11: [

_output_shapes
:1:$\ 

_output_shapes

:11: ]

_output_shapes
:1:$^ 

_output_shapes

:11: _

_output_shapes
:1:$` 

_output_shapes

:1: a

_output_shapes
::b

_output_shapes
: 
??
?
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23584868
m_0__to__m_1_input'
m_0__to__m_1_23584702:1#
m_0__to__m_1_23584704:1'
m_1__to__m_2_23584707:11#
m_1__to__m_2_23584709:1'
m_2__to__m_3_23584712:11#
m_2__to__m_3_23584714:1'
m_3__to__m_4_23584717:11#
m_3__to__m_4_23584719:1'
m_4__to__m_5_23584722:11#
m_4__to__m_5_23584724:1'
m_5__to__m_6_23584727:11#
m_5__to__m_6_23584729:1'
m_6__to__m_7_23584732:11#
m_6__to__m_7_23584734:1'
m_7__to__m_8_23584737:11#
m_7__to__m_8_23584739:1'
m_8__to__m_9_23584742:11#
m_8__to__m_9_23584744:1(
m_9__to__m_10_23584747:11$
m_9__to__m_10_23584749:1)
m_10__to__m_11_23584752:11%
m_10__to__m_11_23584754:1)
m_11__to__m_12_23584757:11%
m_11__to__m_12_23584759:1)
m_12__to__m_13_23584762:11%
m_12__to__m_13_23584764:1)
m_13__to__m_14_23584767:11%
m_13__to__m_14_23584769:1)
m_14__to__m_15_23584772:1%
m_14__to__m_15_23584774:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallm_0__to__m_1_inputm_0__to__m_1_23584702m_0__to__m_1_23584704*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_235837032&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_23584707m_1__to__m_2_23584709*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_235837262&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_23584712m_2__to__m_3_23584714*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_235837492&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_23584717m_3__to__m_4_23584719*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_235837722&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_23584722m_4__to__m_5_23584724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_235837952&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_23584727m_5__to__m_6_23584729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_235838182&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_23584732m_6__to__m_7_23584734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_235838412&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_23584737m_7__to__m_8_23584739*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_235838642&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_23584742m_8__to__m_9_23584744*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_235838872&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_23584747m_9__to__m_10_23584749*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_235839102'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_23584752m_10__to__m_11_23584754*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_235839332(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_23584757m_11__to__m_12_23584759*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_235839562(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_23584762m_12__to__m_13_23584764*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_235839792(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_23584767m_13__to__m_14_23584769*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_235840022(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_23584772m_14__to__m_15_23584774*
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
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_235840252(
&m_14__to__m_15/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_23584702*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_23584707*
_output_shapes

:11*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_23584712*
_output_shapes

:11*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_23584717*
_output_shapes

:11*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_23584722*
_output_shapes

:11*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_23584727*
_output_shapes

:11*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_23584732*
_output_shapes

:11*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_23584737*
_output_shapes

:11*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_23584742*
_output_shapes

:11*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_23584747*
_output_shapes

:11*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112&
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
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_23584752*
_output_shapes

:11*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_23584757*
_output_shapes

:11*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_23584762*
_output_shapes

:11*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_23584767*
_output_shapes

:11*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_23584772*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
%m_14__to__m_15/kernel/Regularizer/mul?
IdentityIdentity/m_14__to__m_15/StatefulPartitionedCall:output:0%^m_0__to__m_1/StatefulPartitionedCall3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp'^m_10__to__m_11/StatefulPartitionedCall5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp'^m_11__to__m_12/StatefulPartitionedCall5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp'^m_12__to__m_13/StatefulPartitionedCall5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp'^m_13__to__m_14/StatefulPartitionedCall5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp'^m_14__to__m_15/StatefulPartitionedCall5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp%^m_1__to__m_2/StatefulPartitionedCall3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp%^m_2__to__m_3/StatefulPartitionedCall3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp%^m_3__to__m_4/StatefulPartitionedCall3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp%^m_4__to__m_5/StatefulPartitionedCall3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp%^m_5__to__m_6/StatefulPartitionedCall3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp%^m_6__to__m_7/StatefulPartitionedCall3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp%^m_7__to__m_8/StatefulPartitionedCall3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp%^m_8__to__m_9/StatefulPartitionedCall3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp&^m_9__to__m_10/StatefulPartitionedCall4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2L
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
?
?
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_23586039

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112&
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
__inference_loss_fn_2_23586241M
;m_2__to__m_3_kernel_regularizer_abs_readvariableop_resource:11
identity??2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_2__to__m_3_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
?
?
__inference_loss_fn_9_23586318N
<m_9__to__m_10_kernel_regularizer_abs_readvariableop_resource:11
identity??3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp<m_9__to__m_10_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112&
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
?
?
__inference_loss_fn_13_23586362O
=m_13__to__m_14_kernel_regularizer_abs_readvariableop_resource:11
identity??4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_13__to__m_14_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
?
?
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_23583726

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_23583887

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
-__inference_L-14-m_0-1_layer_call_fn_23585728

inputs
unknown:1
	unknown_0:1
	unknown_1:11
	unknown_2:1
	unknown_3:11
	unknown_4:1
	unknown_5:11
	unknown_6:1
	unknown_7:11
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1

unknown_13:11

unknown_14:1

unknown_15:11

unknown_16:1

unknown_17:11

unknown_18:1

unknown_19:11

unknown_20:1

unknown_21:11

unknown_22:1

unknown_23:11

unknown_24:1

unknown_25:11

unknown_26:1

unknown_27:1

unknown_28:
identity??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_235845712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_23583818

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_23583864

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_23583772

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_23583703

inputs0
matmul_readvariableop_resource:1-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
:?????????12

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
?
?
__inference_loss_fn_14_23586373O
=m_14__to__m_15_kernel_regularizer_abs_readvariableop_resource:1
identity??4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_14__to__m_15_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_23583910

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112&
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2j
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
1__inference_m_12__to__m_13_layer_call_fn_23586144

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_235839792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_23585879

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
__inference_loss_fn_7_23586296M
;m_7__to__m_8_kernel_regularizer_abs_readvariableop_resource:11
identity??2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_7__to__m_8_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
?
?
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_23585943

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
/__inference_m_6__to__m_7_layer_call_fn_23585952

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_235838412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_23584025

inputs0
matmul_readvariableop_resource:1-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
__inference_loss_fn_12_23586351O
=m_12__to__m_13_kernel_regularizer_abs_readvariableop_resource:11
identity??4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_12__to__m_13_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
1__inference_m_14__to__m_15_layer_call_fn_23586208

inputs
unknown:1
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
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_235840252
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
0__inference_m_9__to__m_10_layer_call_fn_23586048

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_235839102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
??
?
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23585598

inputs=
+m_0__to__m_1_matmul_readvariableop_resource:1:
,m_0__to__m_1_biasadd_readvariableop_resource:1=
+m_1__to__m_2_matmul_readvariableop_resource:11:
,m_1__to__m_2_biasadd_readvariableop_resource:1=
+m_2__to__m_3_matmul_readvariableop_resource:11:
,m_2__to__m_3_biasadd_readvariableop_resource:1=
+m_3__to__m_4_matmul_readvariableop_resource:11:
,m_3__to__m_4_biasadd_readvariableop_resource:1=
+m_4__to__m_5_matmul_readvariableop_resource:11:
,m_4__to__m_5_biasadd_readvariableop_resource:1=
+m_5__to__m_6_matmul_readvariableop_resource:11:
,m_5__to__m_6_biasadd_readvariableop_resource:1=
+m_6__to__m_7_matmul_readvariableop_resource:11:
,m_6__to__m_7_biasadd_readvariableop_resource:1=
+m_7__to__m_8_matmul_readvariableop_resource:11:
,m_7__to__m_8_biasadd_readvariableop_resource:1=
+m_8__to__m_9_matmul_readvariableop_resource:11:
,m_8__to__m_9_biasadd_readvariableop_resource:1>
,m_9__to__m_10_matmul_readvariableop_resource:11;
-m_9__to__m_10_biasadd_readvariableop_resource:1?
-m_10__to__m_11_matmul_readvariableop_resource:11<
.m_10__to__m_11_biasadd_readvariableop_resource:1?
-m_11__to__m_12_matmul_readvariableop_resource:11<
.m_11__to__m_12_biasadd_readvariableop_resource:1?
-m_12__to__m_13_matmul_readvariableop_resource:11<
.m_12__to__m_13_biasadd_readvariableop_resource:1?
-m_13__to__m_14_matmul_readvariableop_resource:11<
.m_13__to__m_14_biasadd_readvariableop_resource:1?
-m_14__to__m_15_matmul_readvariableop_resource:1<
.m_14__to__m_15_biasadd_readvariableop_resource:
identity??#m_0__to__m_1/BiasAdd/ReadVariableOp?"m_0__to__m_1/MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?%m_10__to__m_11/BiasAdd/ReadVariableOp?$m_10__to__m_11/MatMul/ReadVariableOp?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?%m_11__to__m_12/BiasAdd/ReadVariableOp?$m_11__to__m_12/MatMul/ReadVariableOp?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?%m_12__to__m_13/BiasAdd/ReadVariableOp?$m_12__to__m_13/MatMul/ReadVariableOp?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?%m_13__to__m_14/BiasAdd/ReadVariableOp?$m_13__to__m_14/MatMul/ReadVariableOp?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?%m_14__to__m_15/BiasAdd/ReadVariableOp?$m_14__to__m_15/MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?#m_1__to__m_2/BiasAdd/ReadVariableOp?"m_1__to__m_2/MatMul/ReadVariableOp?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?#m_2__to__m_3/BiasAdd/ReadVariableOp?"m_2__to__m_3/MatMul/ReadVariableOp?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?#m_3__to__m_4/BiasAdd/ReadVariableOp?"m_3__to__m_4/MatMul/ReadVariableOp?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?#m_4__to__m_5/BiasAdd/ReadVariableOp?"m_4__to__m_5/MatMul/ReadVariableOp?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?#m_5__to__m_6/BiasAdd/ReadVariableOp?"m_5__to__m_6/MatMul/ReadVariableOp?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?#m_6__to__m_7/BiasAdd/ReadVariableOp?"m_6__to__m_7/MatMul/ReadVariableOp?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?#m_7__to__m_8/BiasAdd/ReadVariableOp?"m_7__to__m_8/MatMul/ReadVariableOp?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?#m_8__to__m_9/BiasAdd/ReadVariableOp?"m_8__to__m_9/MatMul/ReadVariableOp?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?$m_9__to__m_10/BiasAdd/ReadVariableOp?#m_9__to__m_10/MatMul/ReadVariableOp?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
"m_0__to__m_1/MatMul/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:1*
dtype02$
"m_0__to__m_1/MatMul/ReadVariableOp?
m_0__to__m_1/MatMulMatMulinputs*m_0__to__m_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_0__to__m_1/MatMul?
#m_0__to__m_1/BiasAdd/ReadVariableOpReadVariableOp,m_0__to__m_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_0__to__m_1/BiasAdd/ReadVariableOp?
m_0__to__m_1/BiasAddAddm_0__to__m_1/MatMul:product:0+m_0__to__m_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_0__to__m_1/BiasAddz
m_0__to__m_1/ReluRelum_0__to__m_1/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_0__to__m_1/Relu?
"m_1__to__m_2/MatMul/ReadVariableOpReadVariableOp+m_1__to__m_2_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_1__to__m_2/MatMul/ReadVariableOp?
m_1__to__m_2/MatMulMatMulm_0__to__m_1/Relu:activations:0*m_1__to__m_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_1__to__m_2/MatMul?
#m_1__to__m_2/BiasAdd/ReadVariableOpReadVariableOp,m_1__to__m_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_1__to__m_2/BiasAdd/ReadVariableOp?
m_1__to__m_2/BiasAddAddm_1__to__m_2/MatMul:product:0+m_1__to__m_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_1__to__m_2/BiasAddz
m_1__to__m_2/ReluRelum_1__to__m_2/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_1__to__m_2/Relu?
"m_2__to__m_3/MatMul/ReadVariableOpReadVariableOp+m_2__to__m_3_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_2__to__m_3/MatMul/ReadVariableOp?
m_2__to__m_3/MatMulMatMulm_1__to__m_2/Relu:activations:0*m_2__to__m_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_2__to__m_3/MatMul?
#m_2__to__m_3/BiasAdd/ReadVariableOpReadVariableOp,m_2__to__m_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_2__to__m_3/BiasAdd/ReadVariableOp?
m_2__to__m_3/BiasAddAddm_2__to__m_3/MatMul:product:0+m_2__to__m_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_2__to__m_3/BiasAddz
m_2__to__m_3/ReluRelum_2__to__m_3/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_2__to__m_3/Relu?
"m_3__to__m_4/MatMul/ReadVariableOpReadVariableOp+m_3__to__m_4_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_3__to__m_4/MatMul/ReadVariableOp?
m_3__to__m_4/MatMulMatMulm_2__to__m_3/Relu:activations:0*m_3__to__m_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_3__to__m_4/MatMul?
#m_3__to__m_4/BiasAdd/ReadVariableOpReadVariableOp,m_3__to__m_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_3__to__m_4/BiasAdd/ReadVariableOp?
m_3__to__m_4/BiasAddAddm_3__to__m_4/MatMul:product:0+m_3__to__m_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_3__to__m_4/BiasAddz
m_3__to__m_4/ReluRelum_3__to__m_4/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_3__to__m_4/Relu?
"m_4__to__m_5/MatMul/ReadVariableOpReadVariableOp+m_4__to__m_5_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_4__to__m_5/MatMul/ReadVariableOp?
m_4__to__m_5/MatMulMatMulm_3__to__m_4/Relu:activations:0*m_4__to__m_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_4__to__m_5/MatMul?
#m_4__to__m_5/BiasAdd/ReadVariableOpReadVariableOp,m_4__to__m_5_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_4__to__m_5/BiasAdd/ReadVariableOp?
m_4__to__m_5/BiasAddAddm_4__to__m_5/MatMul:product:0+m_4__to__m_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_4__to__m_5/BiasAddz
m_4__to__m_5/ReluRelum_4__to__m_5/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_4__to__m_5/Relu?
"m_5__to__m_6/MatMul/ReadVariableOpReadVariableOp+m_5__to__m_6_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_5__to__m_6/MatMul/ReadVariableOp?
m_5__to__m_6/MatMulMatMulm_4__to__m_5/Relu:activations:0*m_5__to__m_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_5__to__m_6/MatMul?
#m_5__to__m_6/BiasAdd/ReadVariableOpReadVariableOp,m_5__to__m_6_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_5__to__m_6/BiasAdd/ReadVariableOp?
m_5__to__m_6/BiasAddAddm_5__to__m_6/MatMul:product:0+m_5__to__m_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_5__to__m_6/BiasAddz
m_5__to__m_6/ReluRelum_5__to__m_6/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_5__to__m_6/Relu?
"m_6__to__m_7/MatMul/ReadVariableOpReadVariableOp+m_6__to__m_7_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_6__to__m_7/MatMul/ReadVariableOp?
m_6__to__m_7/MatMulMatMulm_5__to__m_6/Relu:activations:0*m_6__to__m_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_6__to__m_7/MatMul?
#m_6__to__m_7/BiasAdd/ReadVariableOpReadVariableOp,m_6__to__m_7_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_6__to__m_7/BiasAdd/ReadVariableOp?
m_6__to__m_7/BiasAddAddm_6__to__m_7/MatMul:product:0+m_6__to__m_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_6__to__m_7/BiasAddz
m_6__to__m_7/ReluRelum_6__to__m_7/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_6__to__m_7/Relu?
"m_7__to__m_8/MatMul/ReadVariableOpReadVariableOp+m_7__to__m_8_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_7__to__m_8/MatMul/ReadVariableOp?
m_7__to__m_8/MatMulMatMulm_6__to__m_7/Relu:activations:0*m_7__to__m_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_7__to__m_8/MatMul?
#m_7__to__m_8/BiasAdd/ReadVariableOpReadVariableOp,m_7__to__m_8_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_7__to__m_8/BiasAdd/ReadVariableOp?
m_7__to__m_8/BiasAddAddm_7__to__m_8/MatMul:product:0+m_7__to__m_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_7__to__m_8/BiasAddz
m_7__to__m_8/ReluRelum_7__to__m_8/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_7__to__m_8/Relu?
"m_8__to__m_9/MatMul/ReadVariableOpReadVariableOp+m_8__to__m_9_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_8__to__m_9/MatMul/ReadVariableOp?
m_8__to__m_9/MatMulMatMulm_7__to__m_8/Relu:activations:0*m_8__to__m_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_8__to__m_9/MatMul?
#m_8__to__m_9/BiasAdd/ReadVariableOpReadVariableOp,m_8__to__m_9_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_8__to__m_9/BiasAdd/ReadVariableOp?
m_8__to__m_9/BiasAddAddm_8__to__m_9/MatMul:product:0+m_8__to__m_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_8__to__m_9/BiasAddz
m_8__to__m_9/ReluRelum_8__to__m_9/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_8__to__m_9/Relu?
#m_9__to__m_10/MatMul/ReadVariableOpReadVariableOp,m_9__to__m_10_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02%
#m_9__to__m_10/MatMul/ReadVariableOp?
m_9__to__m_10/MatMulMatMulm_8__to__m_9/Relu:activations:0+m_9__to__m_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_9__to__m_10/MatMul?
$m_9__to__m_10/BiasAdd/ReadVariableOpReadVariableOp-m_9__to__m_10_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02&
$m_9__to__m_10/BiasAdd/ReadVariableOp?
m_9__to__m_10/BiasAddAddm_9__to__m_10/MatMul:product:0,m_9__to__m_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_9__to__m_10/BiasAdd}
m_9__to__m_10/ReluRelum_9__to__m_10/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_9__to__m_10/Relu?
$m_10__to__m_11/MatMul/ReadVariableOpReadVariableOp-m_10__to__m_11_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02&
$m_10__to__m_11/MatMul/ReadVariableOp?
m_10__to__m_11/MatMulMatMul m_9__to__m_10/Relu:activations:0,m_10__to__m_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_10__to__m_11/MatMul?
%m_10__to__m_11/BiasAdd/ReadVariableOpReadVariableOp.m_10__to__m_11_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02'
%m_10__to__m_11/BiasAdd/ReadVariableOp?
m_10__to__m_11/BiasAddAddm_10__to__m_11/MatMul:product:0-m_10__to__m_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_10__to__m_11/BiasAdd?
m_10__to__m_11/ReluRelum_10__to__m_11/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_10__to__m_11/Relu?
$m_11__to__m_12/MatMul/ReadVariableOpReadVariableOp-m_11__to__m_12_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02&
$m_11__to__m_12/MatMul/ReadVariableOp?
m_11__to__m_12/MatMulMatMul!m_10__to__m_11/Relu:activations:0,m_11__to__m_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_11__to__m_12/MatMul?
%m_11__to__m_12/BiasAdd/ReadVariableOpReadVariableOp.m_11__to__m_12_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02'
%m_11__to__m_12/BiasAdd/ReadVariableOp?
m_11__to__m_12/BiasAddAddm_11__to__m_12/MatMul:product:0-m_11__to__m_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_11__to__m_12/BiasAdd?
m_11__to__m_12/ReluRelum_11__to__m_12/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_11__to__m_12/Relu?
$m_12__to__m_13/MatMul/ReadVariableOpReadVariableOp-m_12__to__m_13_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02&
$m_12__to__m_13/MatMul/ReadVariableOp?
m_12__to__m_13/MatMulMatMul!m_11__to__m_12/Relu:activations:0,m_12__to__m_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_12__to__m_13/MatMul?
%m_12__to__m_13/BiasAdd/ReadVariableOpReadVariableOp.m_12__to__m_13_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02'
%m_12__to__m_13/BiasAdd/ReadVariableOp?
m_12__to__m_13/BiasAddAddm_12__to__m_13/MatMul:product:0-m_12__to__m_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_12__to__m_13/BiasAdd?
m_12__to__m_13/ReluRelum_12__to__m_13/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_12__to__m_13/Relu?
$m_13__to__m_14/MatMul/ReadVariableOpReadVariableOp-m_13__to__m_14_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02&
$m_13__to__m_14/MatMul/ReadVariableOp?
m_13__to__m_14/MatMulMatMul!m_12__to__m_13/Relu:activations:0,m_13__to__m_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_13__to__m_14/MatMul?
%m_13__to__m_14/BiasAdd/ReadVariableOpReadVariableOp.m_13__to__m_14_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02'
%m_13__to__m_14/BiasAdd/ReadVariableOp?
m_13__to__m_14/BiasAddAddm_13__to__m_14/MatMul:product:0-m_13__to__m_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_13__to__m_14/BiasAdd?
m_13__to__m_14/ReluRelum_13__to__m_14/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_13__to__m_14/Relu?
$m_14__to__m_15/MatMul/ReadVariableOpReadVariableOp-m_14__to__m_15_matmul_readvariableop_resource*
_output_shapes

:1*
dtype02&
$m_14__to__m_15/MatMul/ReadVariableOp?
m_14__to__m_15/MatMulMatMul!m_13__to__m_14/Relu:activations:0,m_14__to__m_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/MatMul?
%m_14__to__m_15/BiasAdd/ReadVariableOpReadVariableOp.m_14__to__m_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%m_14__to__m_15/BiasAdd/ReadVariableOp?
m_14__to__m_15/BiasAddAddm_14__to__m_15/MatMul:product:0-m_14__to__m_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/BiasAdd?
m_14__to__m_15/SoftmaxSoftmaxm_14__to__m_15/BiasAdd:z:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/Softmax?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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

:11*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112&
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

:11*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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

:11*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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

:11*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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

:11*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
%m_14__to__m_15/kernel/Regularizer/mul?
IdentityIdentity m_14__to__m_15/Softmax:softmax:0$^m_0__to__m_1/BiasAdd/ReadVariableOp#^m_0__to__m_1/MatMul/ReadVariableOp3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp&^m_10__to__m_11/BiasAdd/ReadVariableOp%^m_10__to__m_11/MatMul/ReadVariableOp5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp&^m_11__to__m_12/BiasAdd/ReadVariableOp%^m_11__to__m_12/MatMul/ReadVariableOp5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp&^m_12__to__m_13/BiasAdd/ReadVariableOp%^m_12__to__m_13/MatMul/ReadVariableOp5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp&^m_13__to__m_14/BiasAdd/ReadVariableOp%^m_13__to__m_14/MatMul/ReadVariableOp5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp&^m_14__to__m_15/BiasAdd/ReadVariableOp%^m_14__to__m_15/MatMul/ReadVariableOp5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp$^m_1__to__m_2/BiasAdd/ReadVariableOp#^m_1__to__m_2/MatMul/ReadVariableOp3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp$^m_2__to__m_3/BiasAdd/ReadVariableOp#^m_2__to__m_3/MatMul/ReadVariableOp3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp$^m_3__to__m_4/BiasAdd/ReadVariableOp#^m_3__to__m_4/MatMul/ReadVariableOp3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp$^m_4__to__m_5/BiasAdd/ReadVariableOp#^m_4__to__m_5/MatMul/ReadVariableOp3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp$^m_5__to__m_6/BiasAdd/ReadVariableOp#^m_5__to__m_6/MatMul/ReadVariableOp3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp$^m_6__to__m_7/BiasAdd/ReadVariableOp#^m_6__to__m_7/MatMul/ReadVariableOp3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp$^m_7__to__m_8/BiasAdd/ReadVariableOp#^m_7__to__m_8/MatMul/ReadVariableOp3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp$^m_8__to__m_9/BiasAdd/ReadVariableOp#^m_8__to__m_9/MatMul/ReadVariableOp3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp%^m_9__to__m_10/BiasAdd/ReadVariableOp$^m_9__to__m_10/MatMul/ReadVariableOp4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2J
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
1__inference_m_11__to__m_12_layer_call_fn_23586112

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_235839562
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_23583979

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
/__inference_m_7__to__m_8_layer_call_fn_23585984

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_235838642
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
-__inference_L-14-m_0-1_layer_call_fn_23584699
m_0__to__m_1_input
unknown:1
	unknown_0:1
	unknown_1:11
	unknown_2:1
	unknown_3:11
	unknown_4:1
	unknown_5:11
	unknown_6:1
	unknown_7:11
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1

unknown_13:11

unknown_14:1

unknown_15:11

unknown_16:1

unknown_17:11

unknown_18:1

unknown_19:11

unknown_20:1

unknown_21:11

unknown_22:1

unknown_23:11

unknown_24:1

unknown_25:11

unknown_26:1

unknown_27:1

unknown_28:
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_235845712
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
__inference_loss_fn_8_23586307M
;m_8__to__m_9_kernel_regularizer_abs_readvariableop_resource:11
identity??2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_8__to__m_9_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
?
?
/__inference_m_1__to__m_2_layer_call_fn_23585792

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_235837262
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
??
?
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23585037
m_0__to__m_1_input'
m_0__to__m_1_23584871:1#
m_0__to__m_1_23584873:1'
m_1__to__m_2_23584876:11#
m_1__to__m_2_23584878:1'
m_2__to__m_3_23584881:11#
m_2__to__m_3_23584883:1'
m_3__to__m_4_23584886:11#
m_3__to__m_4_23584888:1'
m_4__to__m_5_23584891:11#
m_4__to__m_5_23584893:1'
m_5__to__m_6_23584896:11#
m_5__to__m_6_23584898:1'
m_6__to__m_7_23584901:11#
m_6__to__m_7_23584903:1'
m_7__to__m_8_23584906:11#
m_7__to__m_8_23584908:1'
m_8__to__m_9_23584911:11#
m_8__to__m_9_23584913:1(
m_9__to__m_10_23584916:11$
m_9__to__m_10_23584918:1)
m_10__to__m_11_23584921:11%
m_10__to__m_11_23584923:1)
m_11__to__m_12_23584926:11%
m_11__to__m_12_23584928:1)
m_12__to__m_13_23584931:11%
m_12__to__m_13_23584933:1)
m_13__to__m_14_23584936:11%
m_13__to__m_14_23584938:1)
m_14__to__m_15_23584941:1%
m_14__to__m_15_23584943:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallm_0__to__m_1_inputm_0__to__m_1_23584871m_0__to__m_1_23584873*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_235837032&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_23584876m_1__to__m_2_23584878*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_235837262&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_23584881m_2__to__m_3_23584883*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_235837492&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_23584886m_3__to__m_4_23584888*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_235837722&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_23584891m_4__to__m_5_23584893*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_235837952&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_23584896m_5__to__m_6_23584898*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_235838182&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_23584901m_6__to__m_7_23584903*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_235838412&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_23584906m_7__to__m_8_23584908*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_235838642&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_23584911m_8__to__m_9_23584913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_235838872&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_23584916m_9__to__m_10_23584918*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_235839102'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_23584921m_10__to__m_11_23584923*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_235839332(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_23584926m_11__to__m_12_23584928*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_235839562(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_23584931m_12__to__m_13_23584933*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_235839792(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_23584936m_13__to__m_14_23584938*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_235840022(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_23584941m_14__to__m_15_23584943*
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
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_235840252(
&m_14__to__m_15/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_23584871*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_23584876*
_output_shapes

:11*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_23584881*
_output_shapes

:11*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_23584886*
_output_shapes

:11*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_23584891*
_output_shapes

:11*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_23584896*
_output_shapes

:11*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_23584901*
_output_shapes

:11*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_23584906*
_output_shapes

:11*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_23584911*
_output_shapes

:11*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_23584916*
_output_shapes

:11*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112&
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
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_23584921*
_output_shapes

:11*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_23584926*
_output_shapes

:11*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_23584931*
_output_shapes

:11*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_23584936*
_output_shapes

:11*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_23584941*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
%m_14__to__m_15/kernel/Regularizer/mul?
IdentityIdentity/m_14__to__m_15/StatefulPartitionedCall:output:0%^m_0__to__m_1/StatefulPartitionedCall3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp'^m_10__to__m_11/StatefulPartitionedCall5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp'^m_11__to__m_12/StatefulPartitionedCall5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp'^m_12__to__m_13/StatefulPartitionedCall5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp'^m_13__to__m_14/StatefulPartitionedCall5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp'^m_14__to__m_15/StatefulPartitionedCall5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp%^m_1__to__m_2/StatefulPartitionedCall3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp%^m_2__to__m_3/StatefulPartitionedCall3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp%^m_3__to__m_4/StatefulPartitionedCall3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp%^m_4__to__m_5/StatefulPartitionedCall3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp%^m_5__to__m_6/StatefulPartitionedCall3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp%^m_6__to__m_7/StatefulPartitionedCall3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp%^m_7__to__m_8/StatefulPartitionedCall3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp%^m_8__to__m_9/StatefulPartitionedCall3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp&^m_9__to__m_10/StatefulPartitionedCall4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2L
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
??
?>
$__inference__traced_restore_23586988
file_prefix6
$assignvariableop_m_0__to__m_1_kernel:12
$assignvariableop_1_m_0__to__m_1_bias:18
&assignvariableop_2_m_1__to__m_2_kernel:112
$assignvariableop_3_m_1__to__m_2_bias:18
&assignvariableop_4_m_2__to__m_3_kernel:112
$assignvariableop_5_m_2__to__m_3_bias:18
&assignvariableop_6_m_3__to__m_4_kernel:112
$assignvariableop_7_m_3__to__m_4_bias:18
&assignvariableop_8_m_4__to__m_5_kernel:112
$assignvariableop_9_m_4__to__m_5_bias:19
'assignvariableop_10_m_5__to__m_6_kernel:113
%assignvariableop_11_m_5__to__m_6_bias:19
'assignvariableop_12_m_6__to__m_7_kernel:113
%assignvariableop_13_m_6__to__m_7_bias:19
'assignvariableop_14_m_7__to__m_8_kernel:113
%assignvariableop_15_m_7__to__m_8_bias:19
'assignvariableop_16_m_8__to__m_9_kernel:113
%assignvariableop_17_m_8__to__m_9_bias:1:
(assignvariableop_18_m_9__to__m_10_kernel:114
&assignvariableop_19_m_9__to__m_10_bias:1;
)assignvariableop_20_m_10__to__m_11_kernel:115
'assignvariableop_21_m_10__to__m_11_bias:1;
)assignvariableop_22_m_11__to__m_12_kernel:115
'assignvariableop_23_m_11__to__m_12_bias:1;
)assignvariableop_24_m_12__to__m_13_kernel:115
'assignvariableop_25_m_12__to__m_13_bias:1;
)assignvariableop_26_m_13__to__m_14_kernel:115
'assignvariableop_27_m_13__to__m_14_bias:1;
)assignvariableop_28_m_14__to__m_15_kernel:15
'assignvariableop_29_m_14__to__m_15_bias:'
assignvariableop_30_adam_iter:	 )
assignvariableop_31_adam_beta_1: )
assignvariableop_32_adam_beta_2: (
assignvariableop_33_adam_decay: 0
&assignvariableop_34_adam_learning_rate: #
assignvariableop_35_total: #
assignvariableop_36_count: @
.assignvariableop_37_adam_m_0__to__m_1_kernel_m:1:
,assignvariableop_38_adam_m_0__to__m_1_bias_m:1@
.assignvariableop_39_adam_m_1__to__m_2_kernel_m:11:
,assignvariableop_40_adam_m_1__to__m_2_bias_m:1@
.assignvariableop_41_adam_m_2__to__m_3_kernel_m:11:
,assignvariableop_42_adam_m_2__to__m_3_bias_m:1@
.assignvariableop_43_adam_m_3__to__m_4_kernel_m:11:
,assignvariableop_44_adam_m_3__to__m_4_bias_m:1@
.assignvariableop_45_adam_m_4__to__m_5_kernel_m:11:
,assignvariableop_46_adam_m_4__to__m_5_bias_m:1@
.assignvariableop_47_adam_m_5__to__m_6_kernel_m:11:
,assignvariableop_48_adam_m_5__to__m_6_bias_m:1@
.assignvariableop_49_adam_m_6__to__m_7_kernel_m:11:
,assignvariableop_50_adam_m_6__to__m_7_bias_m:1@
.assignvariableop_51_adam_m_7__to__m_8_kernel_m:11:
,assignvariableop_52_adam_m_7__to__m_8_bias_m:1@
.assignvariableop_53_adam_m_8__to__m_9_kernel_m:11:
,assignvariableop_54_adam_m_8__to__m_9_bias_m:1A
/assignvariableop_55_adam_m_9__to__m_10_kernel_m:11;
-assignvariableop_56_adam_m_9__to__m_10_bias_m:1B
0assignvariableop_57_adam_m_10__to__m_11_kernel_m:11<
.assignvariableop_58_adam_m_10__to__m_11_bias_m:1B
0assignvariableop_59_adam_m_11__to__m_12_kernel_m:11<
.assignvariableop_60_adam_m_11__to__m_12_bias_m:1B
0assignvariableop_61_adam_m_12__to__m_13_kernel_m:11<
.assignvariableop_62_adam_m_12__to__m_13_bias_m:1B
0assignvariableop_63_adam_m_13__to__m_14_kernel_m:11<
.assignvariableop_64_adam_m_13__to__m_14_bias_m:1B
0assignvariableop_65_adam_m_14__to__m_15_kernel_m:1<
.assignvariableop_66_adam_m_14__to__m_15_bias_m:@
.assignvariableop_67_adam_m_0__to__m_1_kernel_v:1:
,assignvariableop_68_adam_m_0__to__m_1_bias_v:1@
.assignvariableop_69_adam_m_1__to__m_2_kernel_v:11:
,assignvariableop_70_adam_m_1__to__m_2_bias_v:1@
.assignvariableop_71_adam_m_2__to__m_3_kernel_v:11:
,assignvariableop_72_adam_m_2__to__m_3_bias_v:1@
.assignvariableop_73_adam_m_3__to__m_4_kernel_v:11:
,assignvariableop_74_adam_m_3__to__m_4_bias_v:1@
.assignvariableop_75_adam_m_4__to__m_5_kernel_v:11:
,assignvariableop_76_adam_m_4__to__m_5_bias_v:1@
.assignvariableop_77_adam_m_5__to__m_6_kernel_v:11:
,assignvariableop_78_adam_m_5__to__m_6_bias_v:1@
.assignvariableop_79_adam_m_6__to__m_7_kernel_v:11:
,assignvariableop_80_adam_m_6__to__m_7_bias_v:1@
.assignvariableop_81_adam_m_7__to__m_8_kernel_v:11:
,assignvariableop_82_adam_m_7__to__m_8_bias_v:1@
.assignvariableop_83_adam_m_8__to__m_9_kernel_v:11:
,assignvariableop_84_adam_m_8__to__m_9_bias_v:1A
/assignvariableop_85_adam_m_9__to__m_10_kernel_v:11;
-assignvariableop_86_adam_m_9__to__m_10_bias_v:1B
0assignvariableop_87_adam_m_10__to__m_11_kernel_v:11<
.assignvariableop_88_adam_m_10__to__m_11_bias_v:1B
0assignvariableop_89_adam_m_11__to__m_12_kernel_v:11<
.assignvariableop_90_adam_m_11__to__m_12_bias_v:1B
0assignvariableop_91_adam_m_12__to__m_13_kernel_v:11<
.assignvariableop_92_adam_m_12__to__m_13_bias_v:1B
0assignvariableop_93_adam_m_13__to__m_14_kernel_v:11<
.assignvariableop_94_adam_m_13__to__m_14_bias_v:1B
0assignvariableop_95_adam_m_14__to__m_15_kernel_v:1<
.assignvariableop_96_adam_m_14__to__m_15_bias_v:
identity_98??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_75?AssignVariableOp_76?AssignVariableOp_77?AssignVariableOp_78?AssignVariableOp_79?AssignVariableOp_8?AssignVariableOp_80?AssignVariableOp_81?AssignVariableOp_82?AssignVariableOp_83?AssignVariableOp_84?AssignVariableOp_85?AssignVariableOp_86?AssignVariableOp_87?AssignVariableOp_88?AssignVariableOp_89?AssignVariableOp_9?AssignVariableOp_90?AssignVariableOp_91?AssignVariableOp_92?AssignVariableOp_93?AssignVariableOp_94?AssignVariableOp_95?AssignVariableOp_96?7
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?7
value?6B?6bB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-10/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-10/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-11/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-11/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-12/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-12/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-13/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-13/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBSlayer_with_weights-14/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-14/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:b*
dtype0*?
value?B?bB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*p
dtypesf
d2b	2
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
T0	*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOpassignvariableop_30_adam_iterIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpassignvariableop_31_adam_beta_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOpassignvariableop_32_adam_beta_2Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpassignvariableop_33_adam_decayIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp&assignvariableop_34_adam_learning_rateIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpassignvariableop_35_totalIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpassignvariableop_36_countIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp.assignvariableop_37_adam_m_0__to__m_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp,assignvariableop_38_adam_m_0__to__m_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp.assignvariableop_39_adam_m_1__to__m_2_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp,assignvariableop_40_adam_m_1__to__m_2_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOp.assignvariableop_41_adam_m_2__to__m_3_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOp,assignvariableop_42_adam_m_2__to__m_3_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp.assignvariableop_43_adam_m_3__to__m_4_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp,assignvariableop_44_adam_m_3__to__m_4_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOp.assignvariableop_45_adam_m_4__to__m_5_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp,assignvariableop_46_adam_m_4__to__m_5_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOp.assignvariableop_47_adam_m_5__to__m_6_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp,assignvariableop_48_adam_m_5__to__m_6_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOp.assignvariableop_49_adam_m_6__to__m_7_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOp,assignvariableop_50_adam_m_6__to__m_7_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOp.assignvariableop_51_adam_m_7__to__m_8_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOp,assignvariableop_52_adam_m_7__to__m_8_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOp.assignvariableop_53_adam_m_8__to__m_9_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOp,assignvariableop_54_adam_m_8__to__m_9_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55?
AssignVariableOp_55AssignVariableOp/assignvariableop_55_adam_m_9__to__m_10_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56?
AssignVariableOp_56AssignVariableOp-assignvariableop_56_adam_m_9__to__m_10_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57?
AssignVariableOp_57AssignVariableOp0assignvariableop_57_adam_m_10__to__m_11_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58?
AssignVariableOp_58AssignVariableOp.assignvariableop_58_adam_m_10__to__m_11_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59?
AssignVariableOp_59AssignVariableOp0assignvariableop_59_adam_m_11__to__m_12_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60?
AssignVariableOp_60AssignVariableOp.assignvariableop_60_adam_m_11__to__m_12_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61?
AssignVariableOp_61AssignVariableOp0assignvariableop_61_adam_m_12__to__m_13_kernel_mIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62?
AssignVariableOp_62AssignVariableOp.assignvariableop_62_adam_m_12__to__m_13_bias_mIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63?
AssignVariableOp_63AssignVariableOp0assignvariableop_63_adam_m_13__to__m_14_kernel_mIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_63n
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:2
Identity_64?
AssignVariableOp_64AssignVariableOp.assignvariableop_64_adam_m_13__to__m_14_bias_mIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_64n
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:2
Identity_65?
AssignVariableOp_65AssignVariableOp0assignvariableop_65_adam_m_14__to__m_15_kernel_mIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_65n
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:2
Identity_66?
AssignVariableOp_66AssignVariableOp.assignvariableop_66_adam_m_14__to__m_15_bias_mIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_66n
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:2
Identity_67?
AssignVariableOp_67AssignVariableOp.assignvariableop_67_adam_m_0__to__m_1_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_67n
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:2
Identity_68?
AssignVariableOp_68AssignVariableOp,assignvariableop_68_adam_m_0__to__m_1_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_68n
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:2
Identity_69?
AssignVariableOp_69AssignVariableOp.assignvariableop_69_adam_m_1__to__m_2_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_69n
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:2
Identity_70?
AssignVariableOp_70AssignVariableOp,assignvariableop_70_adam_m_1__to__m_2_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_70n
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:2
Identity_71?
AssignVariableOp_71AssignVariableOp.assignvariableop_71_adam_m_2__to__m_3_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_71n
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:2
Identity_72?
AssignVariableOp_72AssignVariableOp,assignvariableop_72_adam_m_2__to__m_3_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_72n
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:2
Identity_73?
AssignVariableOp_73AssignVariableOp.assignvariableop_73_adam_m_3__to__m_4_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_73n
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:2
Identity_74?
AssignVariableOp_74AssignVariableOp,assignvariableop_74_adam_m_3__to__m_4_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_74n
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:2
Identity_75?
AssignVariableOp_75AssignVariableOp.assignvariableop_75_adam_m_4__to__m_5_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_75n
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:2
Identity_76?
AssignVariableOp_76AssignVariableOp,assignvariableop_76_adam_m_4__to__m_5_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_76n
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:2
Identity_77?
AssignVariableOp_77AssignVariableOp.assignvariableop_77_adam_m_5__to__m_6_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_77n
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:2
Identity_78?
AssignVariableOp_78AssignVariableOp,assignvariableop_78_adam_m_5__to__m_6_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_78n
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:2
Identity_79?
AssignVariableOp_79AssignVariableOp.assignvariableop_79_adam_m_6__to__m_7_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79n
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:2
Identity_80?
AssignVariableOp_80AssignVariableOp,assignvariableop_80_adam_m_6__to__m_7_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_80n
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:2
Identity_81?
AssignVariableOp_81AssignVariableOp.assignvariableop_81_adam_m_7__to__m_8_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_81n
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:2
Identity_82?
AssignVariableOp_82AssignVariableOp,assignvariableop_82_adam_m_7__to__m_8_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_82n
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:2
Identity_83?
AssignVariableOp_83AssignVariableOp.assignvariableop_83_adam_m_8__to__m_9_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_83n
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:2
Identity_84?
AssignVariableOp_84AssignVariableOp,assignvariableop_84_adam_m_8__to__m_9_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_84n
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:2
Identity_85?
AssignVariableOp_85AssignVariableOp/assignvariableop_85_adam_m_9__to__m_10_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_85n
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:2
Identity_86?
AssignVariableOp_86AssignVariableOp-assignvariableop_86_adam_m_9__to__m_10_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_86n
Identity_87IdentityRestoreV2:tensors:87"/device:CPU:0*
T0*
_output_shapes
:2
Identity_87?
AssignVariableOp_87AssignVariableOp0assignvariableop_87_adam_m_10__to__m_11_kernel_vIdentity_87:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_87n
Identity_88IdentityRestoreV2:tensors:88"/device:CPU:0*
T0*
_output_shapes
:2
Identity_88?
AssignVariableOp_88AssignVariableOp.assignvariableop_88_adam_m_10__to__m_11_bias_vIdentity_88:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_88n
Identity_89IdentityRestoreV2:tensors:89"/device:CPU:0*
T0*
_output_shapes
:2
Identity_89?
AssignVariableOp_89AssignVariableOp0assignvariableop_89_adam_m_11__to__m_12_kernel_vIdentity_89:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_89n
Identity_90IdentityRestoreV2:tensors:90"/device:CPU:0*
T0*
_output_shapes
:2
Identity_90?
AssignVariableOp_90AssignVariableOp.assignvariableop_90_adam_m_11__to__m_12_bias_vIdentity_90:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_90n
Identity_91IdentityRestoreV2:tensors:91"/device:CPU:0*
T0*
_output_shapes
:2
Identity_91?
AssignVariableOp_91AssignVariableOp0assignvariableop_91_adam_m_12__to__m_13_kernel_vIdentity_91:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_91n
Identity_92IdentityRestoreV2:tensors:92"/device:CPU:0*
T0*
_output_shapes
:2
Identity_92?
AssignVariableOp_92AssignVariableOp.assignvariableop_92_adam_m_12__to__m_13_bias_vIdentity_92:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_92n
Identity_93IdentityRestoreV2:tensors:93"/device:CPU:0*
T0*
_output_shapes
:2
Identity_93?
AssignVariableOp_93AssignVariableOp0assignvariableop_93_adam_m_13__to__m_14_kernel_vIdentity_93:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_93n
Identity_94IdentityRestoreV2:tensors:94"/device:CPU:0*
T0*
_output_shapes
:2
Identity_94?
AssignVariableOp_94AssignVariableOp.assignvariableop_94_adam_m_13__to__m_14_bias_vIdentity_94:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_94n
Identity_95IdentityRestoreV2:tensors:95"/device:CPU:0*
T0*
_output_shapes
:2
Identity_95?
AssignVariableOp_95AssignVariableOp0assignvariableop_95_adam_m_14__to__m_15_kernel_vIdentity_95:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_95n
Identity_96IdentityRestoreV2:tensors:96"/device:CPU:0*
T0*
_output_shapes
:2
Identity_96?
AssignVariableOp_96AssignVariableOp.assignvariableop_96_adam_m_14__to__m_15_bias_vIdentity_96:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_969
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_97Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_97?
Identity_98IdentityIdentity_97:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_87^AssignVariableOp_88^AssignVariableOp_89^AssignVariableOp_9^AssignVariableOp_90^AssignVariableOp_91^AssignVariableOp_92^AssignVariableOp_93^AssignVariableOp_94^AssignVariableOp_95^AssignVariableOp_96*
T0*
_output_shapes
: 2
Identity_98"#
identity_98Identity_98:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
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
AssignVariableOp_96AssignVariableOp_96:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
/__inference_m_2__to__m_3_layer_call_fn_23585824

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_235837492
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_23586199

inputs0
matmul_readvariableop_resource:1-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
-__inference_L-14-m_0-1_layer_call_fn_23585663

inputs
unknown:1
	unknown_0:1
	unknown_1:11
	unknown_2:1
	unknown_3:11
	unknown_4:1
	unknown_5:11
	unknown_6:1
	unknown_7:11
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1

unknown_13:11

unknown_14:1

unknown_15:11

unknown_16:1

unknown_17:11

unknown_18:1

unknown_19:11

unknown_20:1

unknown_21:11

unknown_22:1

unknown_23:11

unknown_24:1

unknown_25:11

unknown_26:1

unknown_27:1

unknown_28:
identity??StatefulPartitionedCall?
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_235841222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
/__inference_m_8__to__m_9_layer_call_fn_23586016

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_235838872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
__inference_loss_fn_6_23586285M
;m_6__to__m_7_kernel_regularizer_abs_readvariableop_resource:11
identity??2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_6__to__m_7_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
?
?
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_23585847

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
??
?
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23585399

inputs=
+m_0__to__m_1_matmul_readvariableop_resource:1:
,m_0__to__m_1_biasadd_readvariableop_resource:1=
+m_1__to__m_2_matmul_readvariableop_resource:11:
,m_1__to__m_2_biasadd_readvariableop_resource:1=
+m_2__to__m_3_matmul_readvariableop_resource:11:
,m_2__to__m_3_biasadd_readvariableop_resource:1=
+m_3__to__m_4_matmul_readvariableop_resource:11:
,m_3__to__m_4_biasadd_readvariableop_resource:1=
+m_4__to__m_5_matmul_readvariableop_resource:11:
,m_4__to__m_5_biasadd_readvariableop_resource:1=
+m_5__to__m_6_matmul_readvariableop_resource:11:
,m_5__to__m_6_biasadd_readvariableop_resource:1=
+m_6__to__m_7_matmul_readvariableop_resource:11:
,m_6__to__m_7_biasadd_readvariableop_resource:1=
+m_7__to__m_8_matmul_readvariableop_resource:11:
,m_7__to__m_8_biasadd_readvariableop_resource:1=
+m_8__to__m_9_matmul_readvariableop_resource:11:
,m_8__to__m_9_biasadd_readvariableop_resource:1>
,m_9__to__m_10_matmul_readvariableop_resource:11;
-m_9__to__m_10_biasadd_readvariableop_resource:1?
-m_10__to__m_11_matmul_readvariableop_resource:11<
.m_10__to__m_11_biasadd_readvariableop_resource:1?
-m_11__to__m_12_matmul_readvariableop_resource:11<
.m_11__to__m_12_biasadd_readvariableop_resource:1?
-m_12__to__m_13_matmul_readvariableop_resource:11<
.m_12__to__m_13_biasadd_readvariableop_resource:1?
-m_13__to__m_14_matmul_readvariableop_resource:11<
.m_13__to__m_14_biasadd_readvariableop_resource:1?
-m_14__to__m_15_matmul_readvariableop_resource:1<
.m_14__to__m_15_biasadd_readvariableop_resource:
identity??#m_0__to__m_1/BiasAdd/ReadVariableOp?"m_0__to__m_1/MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?%m_10__to__m_11/BiasAdd/ReadVariableOp?$m_10__to__m_11/MatMul/ReadVariableOp?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?%m_11__to__m_12/BiasAdd/ReadVariableOp?$m_11__to__m_12/MatMul/ReadVariableOp?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?%m_12__to__m_13/BiasAdd/ReadVariableOp?$m_12__to__m_13/MatMul/ReadVariableOp?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?%m_13__to__m_14/BiasAdd/ReadVariableOp?$m_13__to__m_14/MatMul/ReadVariableOp?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?%m_14__to__m_15/BiasAdd/ReadVariableOp?$m_14__to__m_15/MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?#m_1__to__m_2/BiasAdd/ReadVariableOp?"m_1__to__m_2/MatMul/ReadVariableOp?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?#m_2__to__m_3/BiasAdd/ReadVariableOp?"m_2__to__m_3/MatMul/ReadVariableOp?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?#m_3__to__m_4/BiasAdd/ReadVariableOp?"m_3__to__m_4/MatMul/ReadVariableOp?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?#m_4__to__m_5/BiasAdd/ReadVariableOp?"m_4__to__m_5/MatMul/ReadVariableOp?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?#m_5__to__m_6/BiasAdd/ReadVariableOp?"m_5__to__m_6/MatMul/ReadVariableOp?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?#m_6__to__m_7/BiasAdd/ReadVariableOp?"m_6__to__m_7/MatMul/ReadVariableOp?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?#m_7__to__m_8/BiasAdd/ReadVariableOp?"m_7__to__m_8/MatMul/ReadVariableOp?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?#m_8__to__m_9/BiasAdd/ReadVariableOp?"m_8__to__m_9/MatMul/ReadVariableOp?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?$m_9__to__m_10/BiasAdd/ReadVariableOp?#m_9__to__m_10/MatMul/ReadVariableOp?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
"m_0__to__m_1/MatMul/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:1*
dtype02$
"m_0__to__m_1/MatMul/ReadVariableOp?
m_0__to__m_1/MatMulMatMulinputs*m_0__to__m_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_0__to__m_1/MatMul?
#m_0__to__m_1/BiasAdd/ReadVariableOpReadVariableOp,m_0__to__m_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_0__to__m_1/BiasAdd/ReadVariableOp?
m_0__to__m_1/BiasAddAddm_0__to__m_1/MatMul:product:0+m_0__to__m_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_0__to__m_1/BiasAddz
m_0__to__m_1/ReluRelum_0__to__m_1/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_0__to__m_1/Relu?
"m_1__to__m_2/MatMul/ReadVariableOpReadVariableOp+m_1__to__m_2_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_1__to__m_2/MatMul/ReadVariableOp?
m_1__to__m_2/MatMulMatMulm_0__to__m_1/Relu:activations:0*m_1__to__m_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_1__to__m_2/MatMul?
#m_1__to__m_2/BiasAdd/ReadVariableOpReadVariableOp,m_1__to__m_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_1__to__m_2/BiasAdd/ReadVariableOp?
m_1__to__m_2/BiasAddAddm_1__to__m_2/MatMul:product:0+m_1__to__m_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_1__to__m_2/BiasAddz
m_1__to__m_2/ReluRelum_1__to__m_2/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_1__to__m_2/Relu?
"m_2__to__m_3/MatMul/ReadVariableOpReadVariableOp+m_2__to__m_3_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_2__to__m_3/MatMul/ReadVariableOp?
m_2__to__m_3/MatMulMatMulm_1__to__m_2/Relu:activations:0*m_2__to__m_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_2__to__m_3/MatMul?
#m_2__to__m_3/BiasAdd/ReadVariableOpReadVariableOp,m_2__to__m_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_2__to__m_3/BiasAdd/ReadVariableOp?
m_2__to__m_3/BiasAddAddm_2__to__m_3/MatMul:product:0+m_2__to__m_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_2__to__m_3/BiasAddz
m_2__to__m_3/ReluRelum_2__to__m_3/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_2__to__m_3/Relu?
"m_3__to__m_4/MatMul/ReadVariableOpReadVariableOp+m_3__to__m_4_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_3__to__m_4/MatMul/ReadVariableOp?
m_3__to__m_4/MatMulMatMulm_2__to__m_3/Relu:activations:0*m_3__to__m_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_3__to__m_4/MatMul?
#m_3__to__m_4/BiasAdd/ReadVariableOpReadVariableOp,m_3__to__m_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_3__to__m_4/BiasAdd/ReadVariableOp?
m_3__to__m_4/BiasAddAddm_3__to__m_4/MatMul:product:0+m_3__to__m_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_3__to__m_4/BiasAddz
m_3__to__m_4/ReluRelum_3__to__m_4/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_3__to__m_4/Relu?
"m_4__to__m_5/MatMul/ReadVariableOpReadVariableOp+m_4__to__m_5_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_4__to__m_5/MatMul/ReadVariableOp?
m_4__to__m_5/MatMulMatMulm_3__to__m_4/Relu:activations:0*m_4__to__m_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_4__to__m_5/MatMul?
#m_4__to__m_5/BiasAdd/ReadVariableOpReadVariableOp,m_4__to__m_5_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_4__to__m_5/BiasAdd/ReadVariableOp?
m_4__to__m_5/BiasAddAddm_4__to__m_5/MatMul:product:0+m_4__to__m_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_4__to__m_5/BiasAddz
m_4__to__m_5/ReluRelum_4__to__m_5/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_4__to__m_5/Relu?
"m_5__to__m_6/MatMul/ReadVariableOpReadVariableOp+m_5__to__m_6_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_5__to__m_6/MatMul/ReadVariableOp?
m_5__to__m_6/MatMulMatMulm_4__to__m_5/Relu:activations:0*m_5__to__m_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_5__to__m_6/MatMul?
#m_5__to__m_6/BiasAdd/ReadVariableOpReadVariableOp,m_5__to__m_6_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_5__to__m_6/BiasAdd/ReadVariableOp?
m_5__to__m_6/BiasAddAddm_5__to__m_6/MatMul:product:0+m_5__to__m_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_5__to__m_6/BiasAddz
m_5__to__m_6/ReluRelum_5__to__m_6/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_5__to__m_6/Relu?
"m_6__to__m_7/MatMul/ReadVariableOpReadVariableOp+m_6__to__m_7_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_6__to__m_7/MatMul/ReadVariableOp?
m_6__to__m_7/MatMulMatMulm_5__to__m_6/Relu:activations:0*m_6__to__m_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_6__to__m_7/MatMul?
#m_6__to__m_7/BiasAdd/ReadVariableOpReadVariableOp,m_6__to__m_7_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_6__to__m_7/BiasAdd/ReadVariableOp?
m_6__to__m_7/BiasAddAddm_6__to__m_7/MatMul:product:0+m_6__to__m_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_6__to__m_7/BiasAddz
m_6__to__m_7/ReluRelum_6__to__m_7/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_6__to__m_7/Relu?
"m_7__to__m_8/MatMul/ReadVariableOpReadVariableOp+m_7__to__m_8_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_7__to__m_8/MatMul/ReadVariableOp?
m_7__to__m_8/MatMulMatMulm_6__to__m_7/Relu:activations:0*m_7__to__m_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_7__to__m_8/MatMul?
#m_7__to__m_8/BiasAdd/ReadVariableOpReadVariableOp,m_7__to__m_8_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_7__to__m_8/BiasAdd/ReadVariableOp?
m_7__to__m_8/BiasAddAddm_7__to__m_8/MatMul:product:0+m_7__to__m_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_7__to__m_8/BiasAddz
m_7__to__m_8/ReluRelum_7__to__m_8/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_7__to__m_8/Relu?
"m_8__to__m_9/MatMul/ReadVariableOpReadVariableOp+m_8__to__m_9_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02$
"m_8__to__m_9/MatMul/ReadVariableOp?
m_8__to__m_9/MatMulMatMulm_7__to__m_8/Relu:activations:0*m_8__to__m_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_8__to__m_9/MatMul?
#m_8__to__m_9/BiasAdd/ReadVariableOpReadVariableOp,m_8__to__m_9_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02%
#m_8__to__m_9/BiasAdd/ReadVariableOp?
m_8__to__m_9/BiasAddAddm_8__to__m_9/MatMul:product:0+m_8__to__m_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_8__to__m_9/BiasAddz
m_8__to__m_9/ReluRelum_8__to__m_9/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_8__to__m_9/Relu?
#m_9__to__m_10/MatMul/ReadVariableOpReadVariableOp,m_9__to__m_10_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02%
#m_9__to__m_10/MatMul/ReadVariableOp?
m_9__to__m_10/MatMulMatMulm_8__to__m_9/Relu:activations:0+m_9__to__m_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_9__to__m_10/MatMul?
$m_9__to__m_10/BiasAdd/ReadVariableOpReadVariableOp-m_9__to__m_10_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02&
$m_9__to__m_10/BiasAdd/ReadVariableOp?
m_9__to__m_10/BiasAddAddm_9__to__m_10/MatMul:product:0,m_9__to__m_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_9__to__m_10/BiasAdd}
m_9__to__m_10/ReluRelum_9__to__m_10/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_9__to__m_10/Relu?
$m_10__to__m_11/MatMul/ReadVariableOpReadVariableOp-m_10__to__m_11_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02&
$m_10__to__m_11/MatMul/ReadVariableOp?
m_10__to__m_11/MatMulMatMul m_9__to__m_10/Relu:activations:0,m_10__to__m_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_10__to__m_11/MatMul?
%m_10__to__m_11/BiasAdd/ReadVariableOpReadVariableOp.m_10__to__m_11_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02'
%m_10__to__m_11/BiasAdd/ReadVariableOp?
m_10__to__m_11/BiasAddAddm_10__to__m_11/MatMul:product:0-m_10__to__m_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_10__to__m_11/BiasAdd?
m_10__to__m_11/ReluRelum_10__to__m_11/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_10__to__m_11/Relu?
$m_11__to__m_12/MatMul/ReadVariableOpReadVariableOp-m_11__to__m_12_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02&
$m_11__to__m_12/MatMul/ReadVariableOp?
m_11__to__m_12/MatMulMatMul!m_10__to__m_11/Relu:activations:0,m_11__to__m_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_11__to__m_12/MatMul?
%m_11__to__m_12/BiasAdd/ReadVariableOpReadVariableOp.m_11__to__m_12_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02'
%m_11__to__m_12/BiasAdd/ReadVariableOp?
m_11__to__m_12/BiasAddAddm_11__to__m_12/MatMul:product:0-m_11__to__m_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_11__to__m_12/BiasAdd?
m_11__to__m_12/ReluRelum_11__to__m_12/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_11__to__m_12/Relu?
$m_12__to__m_13/MatMul/ReadVariableOpReadVariableOp-m_12__to__m_13_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02&
$m_12__to__m_13/MatMul/ReadVariableOp?
m_12__to__m_13/MatMulMatMul!m_11__to__m_12/Relu:activations:0,m_12__to__m_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_12__to__m_13/MatMul?
%m_12__to__m_13/BiasAdd/ReadVariableOpReadVariableOp.m_12__to__m_13_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02'
%m_12__to__m_13/BiasAdd/ReadVariableOp?
m_12__to__m_13/BiasAddAddm_12__to__m_13/MatMul:product:0-m_12__to__m_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_12__to__m_13/BiasAdd?
m_12__to__m_13/ReluRelum_12__to__m_13/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_12__to__m_13/Relu?
$m_13__to__m_14/MatMul/ReadVariableOpReadVariableOp-m_13__to__m_14_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02&
$m_13__to__m_14/MatMul/ReadVariableOp?
m_13__to__m_14/MatMulMatMul!m_12__to__m_13/Relu:activations:0,m_13__to__m_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_13__to__m_14/MatMul?
%m_13__to__m_14/BiasAdd/ReadVariableOpReadVariableOp.m_13__to__m_14_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype02'
%m_13__to__m_14/BiasAdd/ReadVariableOp?
m_13__to__m_14/BiasAddAddm_13__to__m_14/MatMul:product:0-m_13__to__m_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
m_13__to__m_14/BiasAdd?
m_13__to__m_14/ReluRelum_13__to__m_14/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
m_13__to__m_14/Relu?
$m_14__to__m_15/MatMul/ReadVariableOpReadVariableOp-m_14__to__m_15_matmul_readvariableop_resource*
_output_shapes

:1*
dtype02&
$m_14__to__m_15/MatMul/ReadVariableOp?
m_14__to__m_15/MatMulMatMul!m_13__to__m_14/Relu:activations:0,m_14__to__m_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/MatMul?
%m_14__to__m_15/BiasAdd/ReadVariableOpReadVariableOp.m_14__to__m_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%m_14__to__m_15/BiasAdd/ReadVariableOp?
m_14__to__m_15/BiasAddAddm_14__to__m_15/MatMul:product:0-m_14__to__m_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/BiasAdd?
m_14__to__m_15/SoftmaxSoftmaxm_14__to__m_15/BiasAdd:z:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/Softmax?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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

:11*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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

:11*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112&
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

:11*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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

:11*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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

:11*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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

:11*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
%m_14__to__m_15/kernel/Regularizer/mul?
IdentityIdentity m_14__to__m_15/Softmax:softmax:0$^m_0__to__m_1/BiasAdd/ReadVariableOp#^m_0__to__m_1/MatMul/ReadVariableOp3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp&^m_10__to__m_11/BiasAdd/ReadVariableOp%^m_10__to__m_11/MatMul/ReadVariableOp5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp&^m_11__to__m_12/BiasAdd/ReadVariableOp%^m_11__to__m_12/MatMul/ReadVariableOp5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp&^m_12__to__m_13/BiasAdd/ReadVariableOp%^m_12__to__m_13/MatMul/ReadVariableOp5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp&^m_13__to__m_14/BiasAdd/ReadVariableOp%^m_13__to__m_14/MatMul/ReadVariableOp5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp&^m_14__to__m_15/BiasAdd/ReadVariableOp%^m_14__to__m_15/MatMul/ReadVariableOp5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp$^m_1__to__m_2/BiasAdd/ReadVariableOp#^m_1__to__m_2/MatMul/ReadVariableOp3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp$^m_2__to__m_3/BiasAdd/ReadVariableOp#^m_2__to__m_3/MatMul/ReadVariableOp3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp$^m_3__to__m_4/BiasAdd/ReadVariableOp#^m_3__to__m_4/MatMul/ReadVariableOp3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp$^m_4__to__m_5/BiasAdd/ReadVariableOp#^m_4__to__m_5/MatMul/ReadVariableOp3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp$^m_5__to__m_6/BiasAdd/ReadVariableOp#^m_5__to__m_6/MatMul/ReadVariableOp3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp$^m_6__to__m_7/BiasAdd/ReadVariableOp#^m_6__to__m_7/MatMul/ReadVariableOp3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp$^m_7__to__m_8/BiasAdd/ReadVariableOp#^m_7__to__m_8/MatMul/ReadVariableOp3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp$^m_8__to__m_9/BiasAdd/ReadVariableOp#^m_8__to__m_9/MatMul/ReadVariableOp3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp%^m_9__to__m_10/BiasAdd/ReadVariableOp$^m_9__to__m_10/MatMul/ReadVariableOp4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2J
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
?
?
__inference_loss_fn_5_23586274M
;m_5__to__m_6_kernel_regularizer_abs_readvariableop_resource:11
identity??2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_5__to__m_6_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
?
?
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_23585911

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_23585975

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_23583956

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
/__inference_m_4__to__m_5_layer_call_fn_23585888

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_235837952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_23586007

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_23586071

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_23583795

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
??
?
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23584571

inputs'
m_0__to__m_1_23584405:1#
m_0__to__m_1_23584407:1'
m_1__to__m_2_23584410:11#
m_1__to__m_2_23584412:1'
m_2__to__m_3_23584415:11#
m_2__to__m_3_23584417:1'
m_3__to__m_4_23584420:11#
m_3__to__m_4_23584422:1'
m_4__to__m_5_23584425:11#
m_4__to__m_5_23584427:1'
m_5__to__m_6_23584430:11#
m_5__to__m_6_23584432:1'
m_6__to__m_7_23584435:11#
m_6__to__m_7_23584437:1'
m_7__to__m_8_23584440:11#
m_7__to__m_8_23584442:1'
m_8__to__m_9_23584445:11#
m_8__to__m_9_23584447:1(
m_9__to__m_10_23584450:11$
m_9__to__m_10_23584452:1)
m_10__to__m_11_23584455:11%
m_10__to__m_11_23584457:1)
m_11__to__m_12_23584460:11%
m_11__to__m_12_23584462:1)
m_12__to__m_13_23584465:11%
m_12__to__m_13_23584467:1)
m_13__to__m_14_23584470:11%
m_13__to__m_14_23584472:1)
m_14__to__m_15_23584475:1%
m_14__to__m_15_23584477:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallinputsm_0__to__m_1_23584405m_0__to__m_1_23584407*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_235837032&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_23584410m_1__to__m_2_23584412*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_235837262&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_23584415m_2__to__m_3_23584417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_235837492&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_23584420m_3__to__m_4_23584422*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_235837722&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_23584425m_4__to__m_5_23584427*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_235837952&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_23584430m_5__to__m_6_23584432*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_235838182&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_23584435m_6__to__m_7_23584437*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_235838412&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_23584440m_7__to__m_8_23584442*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_235838642&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_23584445m_8__to__m_9_23584447*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_235838872&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_23584450m_9__to__m_10_23584452*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_235839102'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_23584455m_10__to__m_11_23584457*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_235839332(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_23584460m_11__to__m_12_23584462*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_235839562(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_23584465m_12__to__m_13_23584467*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_235839792(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_23584470m_13__to__m_14_23584472*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_235840022(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_23584475m_14__to__m_15_23584477*
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
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_235840252(
&m_14__to__m_15/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_23584405*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_23584410*
_output_shapes

:11*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_23584415*
_output_shapes

:11*
dtype024
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?
#m_2__to__m_3/kernel/Regularizer/AbsAbs:m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_23584420*
_output_shapes

:11*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_23584425*
_output_shapes

:11*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_23584430*
_output_shapes

:11*
dtype024
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?
#m_5__to__m_6/kernel/Regularizer/AbsAbs:m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_23584435*
_output_shapes

:11*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_23584440*
_output_shapes

:11*
dtype024
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?
#m_7__to__m_8/kernel/Regularizer/AbsAbs:m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_23584445*
_output_shapes

:11*
dtype024
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?
#m_8__to__m_9/kernel/Regularizer/AbsAbs:m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_23584450*
_output_shapes

:11*
dtype025
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_9__to__m_10/kernel/Regularizer/AbsAbs;m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112&
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
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_23584455*
_output_shapes

:11*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_23584460*
_output_shapes

:11*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_23584465*
_output_shapes

:11*
dtype026
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?
%m_12__to__m_13/kernel/Regularizer/AbsAbs<m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_23584470*
_output_shapes

:11*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_23584475*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
%m_14__to__m_15/kernel/Regularizer/mul?
IdentityIdentity/m_14__to__m_15/StatefulPartitionedCall:output:0%^m_0__to__m_1/StatefulPartitionedCall3^m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp'^m_10__to__m_11/StatefulPartitionedCall5^m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp'^m_11__to__m_12/StatefulPartitionedCall5^m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp'^m_12__to__m_13/StatefulPartitionedCall5^m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp'^m_13__to__m_14/StatefulPartitionedCall5^m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp'^m_14__to__m_15/StatefulPartitionedCall5^m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp%^m_1__to__m_2/StatefulPartitionedCall3^m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp%^m_2__to__m_3/StatefulPartitionedCall3^m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp%^m_3__to__m_4/StatefulPartitionedCall3^m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp%^m_4__to__m_5/StatefulPartitionedCall3^m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp%^m_5__to__m_6/StatefulPartitionedCall3^m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp%^m_6__to__m_7/StatefulPartitionedCall3^m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp%^m_7__to__m_8/StatefulPartitionedCall3^m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp%^m_8__to__m_9/StatefulPartitionedCall3^m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp&^m_9__to__m_10/StatefulPartitionedCall4^m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp2L
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
?
?
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_23583933

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_23586230M
;m_1__to__m_2_kernel_regularizer_abs_readvariableop_resource:11
identity??2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_1__to__m_2_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
?
?
-__inference_L-14-m_0-1_layer_call_fn_23584185
m_0__to__m_1_input
unknown:1
	unknown_0:1
	unknown_1:11
	unknown_2:1
	unknown_3:11
	unknown_4:1
	unknown_5:11
	unknown_6:1
	unknown_7:11
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1

unknown_13:11

unknown_14:1

unknown_15:11

unknown_16:1

unknown_17:11

unknown_18:1

unknown_19:11

unknown_20:1

unknown_21:11

unknown_22:1

unknown_23:11

unknown_24:1

unknown_25:11

unknown_26:1

unknown_27:1

unknown_28:
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_235841222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
__inference_loss_fn_3_23586252M
;m_3__to__m_4_kernel_regularizer_abs_readvariableop_resource:11
identity??2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_3__to__m_4_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?
#m_3__to__m_4/kernel/Regularizer/AbsAbs:m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
?
?
1__inference_m_10__to__m_11_layer_call_fn_23586080

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_235839332
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_23583841

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?
#m_6__to__m_7/kernel/Regularizer/AbsAbs:m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
__inference_loss_fn_11_23586340O
=m_11__to__m_12_kernel_regularizer_abs_readvariableop_resource:11
identity??4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_11__to__m_12_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
?
?
&__inference_signature_wrapper_23585200
m_0__to__m_1_input
unknown:1
	unknown_0:1
	unknown_1:11
	unknown_2:1
	unknown_3:11
	unknown_4:1
	unknown_5:11
	unknown_6:1
	unknown_7:11
	unknown_8:1
	unknown_9:11

unknown_10:1

unknown_11:11

unknown_12:1

unknown_13:11

unknown_14:1

unknown_15:11

unknown_16:1

unknown_17:11

unknown_18:1

unknown_19:11

unknown_20:1

unknown_21:11

unknown_22:1

unknown_23:11

unknown_24:1

unknown_25:11

unknown_26:1

unknown_27:1

unknown_28:
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
unknown_28**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_235836792
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_23585751

inputs0
matmul_readvariableop_resource:1-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
:?????????12

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
/__inference_m_0__to__m_1_layer_call_fn_23585760

inputs
unknown:1
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_235837032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

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
?
?
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_23586103

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?
%m_11__to__m_12/kernel/Regularizer/AbsAbs<m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
__inference_loss_fn_10_23586329O
=m_10__to__m_11_kernel_regularizer_abs_readvariableop_resource:11
identity??4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_10__to__m_11_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?
%m_10__to__m_11/kernel/Regularizer/AbsAbs<m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_23585783

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?
#m_1__to__m_2/kernel/Regularizer/AbsAbs:m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
/__inference_m_5__to__m_6_layer_call_fn_23585920

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_235838182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
1__inference_m_13__to__m_14_layer_call_fn_23586176

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_235840022
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
__inference_loss_fn_4_23586263M
;m_4__to__m_5_kernel_regularizer_abs_readvariableop_resource:11
identity??2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_4__to__m_5_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:11*
dtype024
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?
#m_4__to__m_5/kernel/Regularizer/AbsAbs:m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112%
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
??
?
#__inference__wrapped_model_23583679
m_0__to__m_1_inputH
6l_14_m_0_1_m_0__to__m_1_matmul_readvariableop_resource:1E
7l_14_m_0_1_m_0__to__m_1_biasadd_readvariableop_resource:1H
6l_14_m_0_1_m_1__to__m_2_matmul_readvariableop_resource:11E
7l_14_m_0_1_m_1__to__m_2_biasadd_readvariableop_resource:1H
6l_14_m_0_1_m_2__to__m_3_matmul_readvariableop_resource:11E
7l_14_m_0_1_m_2__to__m_3_biasadd_readvariableop_resource:1H
6l_14_m_0_1_m_3__to__m_4_matmul_readvariableop_resource:11E
7l_14_m_0_1_m_3__to__m_4_biasadd_readvariableop_resource:1H
6l_14_m_0_1_m_4__to__m_5_matmul_readvariableop_resource:11E
7l_14_m_0_1_m_4__to__m_5_biasadd_readvariableop_resource:1H
6l_14_m_0_1_m_5__to__m_6_matmul_readvariableop_resource:11E
7l_14_m_0_1_m_5__to__m_6_biasadd_readvariableop_resource:1H
6l_14_m_0_1_m_6__to__m_7_matmul_readvariableop_resource:11E
7l_14_m_0_1_m_6__to__m_7_biasadd_readvariableop_resource:1H
6l_14_m_0_1_m_7__to__m_8_matmul_readvariableop_resource:11E
7l_14_m_0_1_m_7__to__m_8_biasadd_readvariableop_resource:1H
6l_14_m_0_1_m_8__to__m_9_matmul_readvariableop_resource:11E
7l_14_m_0_1_m_8__to__m_9_biasadd_readvariableop_resource:1I
7l_14_m_0_1_m_9__to__m_10_matmul_readvariableop_resource:11F
8l_14_m_0_1_m_9__to__m_10_biasadd_readvariableop_resource:1J
8l_14_m_0_1_m_10__to__m_11_matmul_readvariableop_resource:11G
9l_14_m_0_1_m_10__to__m_11_biasadd_readvariableop_resource:1J
8l_14_m_0_1_m_11__to__m_12_matmul_readvariableop_resource:11G
9l_14_m_0_1_m_11__to__m_12_biasadd_readvariableop_resource:1J
8l_14_m_0_1_m_12__to__m_13_matmul_readvariableop_resource:11G
9l_14_m_0_1_m_12__to__m_13_biasadd_readvariableop_resource:1J
8l_14_m_0_1_m_13__to__m_14_matmul_readvariableop_resource:11G
9l_14_m_0_1_m_13__to__m_14_biasadd_readvariableop_resource:1J
8l_14_m_0_1_m_14__to__m_15_matmul_readvariableop_resource:1G
9l_14_m_0_1_m_14__to__m_15_biasadd_readvariableop_resource:
identity??.L-14-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp?-L-14-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp?0L-14-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp?/L-14-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp?0L-14-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp?/L-14-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp?0L-14-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp?/L-14-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp?0L-14-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp?/L-14-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp?0L-14-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp?/L-14-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp?.L-14-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp?-L-14-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp?.L-14-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp?-L-14-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp?.L-14-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp?-L-14-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp?.L-14-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp?-L-14-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp?.L-14-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp?-L-14-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp?.L-14-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp?-L-14-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp?.L-14-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp?-L-14-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp?.L-14-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp?-L-14-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp?/L-14-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp?.L-14-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp?
-L-14-m_0-1/m_0__to__m_1/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_1_m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:1*
dtype02/
-L-14-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp?
L-14-m_0-1/m_0__to__m_1/MatMulMatMulm_0__to__m_1_input5L-14-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_0__to__m_1/MatMul?
.L-14-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_1_m_0__to__m_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp?
L-14-m_0-1/m_0__to__m_1/BiasAddAdd(L-14-m_0-1/m_0__to__m_1/MatMul:product:06L-14-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-1/m_0__to__m_1/BiasAdd?
L-14-m_0-1/m_0__to__m_1/ReluRelu#L-14-m_0-1/m_0__to__m_1/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-1/m_0__to__m_1/Relu?
-L-14-m_0-1/m_1__to__m_2/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_1_m_1__to__m_2_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp?
L-14-m_0-1/m_1__to__m_2/MatMulMatMul*L-14-m_0-1/m_0__to__m_1/Relu:activations:05L-14-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_1__to__m_2/MatMul?
.L-14-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_1_m_1__to__m_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp?
L-14-m_0-1/m_1__to__m_2/BiasAddAdd(L-14-m_0-1/m_1__to__m_2/MatMul:product:06L-14-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-1/m_1__to__m_2/BiasAdd?
L-14-m_0-1/m_1__to__m_2/ReluRelu#L-14-m_0-1/m_1__to__m_2/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-1/m_1__to__m_2/Relu?
-L-14-m_0-1/m_2__to__m_3/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_1_m_2__to__m_3_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp?
L-14-m_0-1/m_2__to__m_3/MatMulMatMul*L-14-m_0-1/m_1__to__m_2/Relu:activations:05L-14-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_2__to__m_3/MatMul?
.L-14-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_1_m_2__to__m_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp?
L-14-m_0-1/m_2__to__m_3/BiasAddAdd(L-14-m_0-1/m_2__to__m_3/MatMul:product:06L-14-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-1/m_2__to__m_3/BiasAdd?
L-14-m_0-1/m_2__to__m_3/ReluRelu#L-14-m_0-1/m_2__to__m_3/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-1/m_2__to__m_3/Relu?
-L-14-m_0-1/m_3__to__m_4/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_1_m_3__to__m_4_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp?
L-14-m_0-1/m_3__to__m_4/MatMulMatMul*L-14-m_0-1/m_2__to__m_3/Relu:activations:05L-14-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_3__to__m_4/MatMul?
.L-14-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_1_m_3__to__m_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp?
L-14-m_0-1/m_3__to__m_4/BiasAddAdd(L-14-m_0-1/m_3__to__m_4/MatMul:product:06L-14-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-1/m_3__to__m_4/BiasAdd?
L-14-m_0-1/m_3__to__m_4/ReluRelu#L-14-m_0-1/m_3__to__m_4/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-1/m_3__to__m_4/Relu?
-L-14-m_0-1/m_4__to__m_5/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_1_m_4__to__m_5_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp?
L-14-m_0-1/m_4__to__m_5/MatMulMatMul*L-14-m_0-1/m_3__to__m_4/Relu:activations:05L-14-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_4__to__m_5/MatMul?
.L-14-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_1_m_4__to__m_5_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp?
L-14-m_0-1/m_4__to__m_5/BiasAddAdd(L-14-m_0-1/m_4__to__m_5/MatMul:product:06L-14-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-1/m_4__to__m_5/BiasAdd?
L-14-m_0-1/m_4__to__m_5/ReluRelu#L-14-m_0-1/m_4__to__m_5/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-1/m_4__to__m_5/Relu?
-L-14-m_0-1/m_5__to__m_6/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_1_m_5__to__m_6_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp?
L-14-m_0-1/m_5__to__m_6/MatMulMatMul*L-14-m_0-1/m_4__to__m_5/Relu:activations:05L-14-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_5__to__m_6/MatMul?
.L-14-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_1_m_5__to__m_6_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp?
L-14-m_0-1/m_5__to__m_6/BiasAddAdd(L-14-m_0-1/m_5__to__m_6/MatMul:product:06L-14-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-1/m_5__to__m_6/BiasAdd?
L-14-m_0-1/m_5__to__m_6/ReluRelu#L-14-m_0-1/m_5__to__m_6/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-1/m_5__to__m_6/Relu?
-L-14-m_0-1/m_6__to__m_7/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_1_m_6__to__m_7_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp?
L-14-m_0-1/m_6__to__m_7/MatMulMatMul*L-14-m_0-1/m_5__to__m_6/Relu:activations:05L-14-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_6__to__m_7/MatMul?
.L-14-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_1_m_6__to__m_7_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp?
L-14-m_0-1/m_6__to__m_7/BiasAddAdd(L-14-m_0-1/m_6__to__m_7/MatMul:product:06L-14-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-1/m_6__to__m_7/BiasAdd?
L-14-m_0-1/m_6__to__m_7/ReluRelu#L-14-m_0-1/m_6__to__m_7/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-1/m_6__to__m_7/Relu?
-L-14-m_0-1/m_7__to__m_8/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_1_m_7__to__m_8_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp?
L-14-m_0-1/m_7__to__m_8/MatMulMatMul*L-14-m_0-1/m_6__to__m_7/Relu:activations:05L-14-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_7__to__m_8/MatMul?
.L-14-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_1_m_7__to__m_8_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp?
L-14-m_0-1/m_7__to__m_8/BiasAddAdd(L-14-m_0-1/m_7__to__m_8/MatMul:product:06L-14-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-1/m_7__to__m_8/BiasAdd?
L-14-m_0-1/m_7__to__m_8/ReluRelu#L-14-m_0-1/m_7__to__m_8/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-1/m_7__to__m_8/Relu?
-L-14-m_0-1/m_8__to__m_9/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_1_m_8__to__m_9_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp?
L-14-m_0-1/m_8__to__m_9/MatMulMatMul*L-14-m_0-1/m_7__to__m_8/Relu:activations:05L-14-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_8__to__m_9/MatMul?
.L-14-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_1_m_8__to__m_9_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp?
L-14-m_0-1/m_8__to__m_9/BiasAddAdd(L-14-m_0-1/m_8__to__m_9/MatMul:product:06L-14-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-1/m_8__to__m_9/BiasAdd?
L-14-m_0-1/m_8__to__m_9/ReluRelu#L-14-m_0-1/m_8__to__m_9/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-1/m_8__to__m_9/Relu?
.L-14-m_0-1/m_9__to__m_10/MatMul/ReadVariableOpReadVariableOp7l_14_m_0_1_m_9__to__m_10_matmul_readvariableop_resource*
_output_shapes

:11*
dtype020
.L-14-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp?
L-14-m_0-1/m_9__to__m_10/MatMulMatMul*L-14-m_0-1/m_8__to__m_9/Relu:activations:06L-14-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-1/m_9__to__m_10/MatMul?
/L-14-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOpReadVariableOp8l_14_m_0_1_m_9__to__m_10_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype021
/L-14-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp?
 L-14-m_0-1/m_9__to__m_10/BiasAddAdd)L-14-m_0-1/m_9__to__m_10/MatMul:product:07L-14-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12"
 L-14-m_0-1/m_9__to__m_10/BiasAdd?
L-14-m_0-1/m_9__to__m_10/ReluRelu$L-14-m_0-1/m_9__to__m_10/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-1/m_9__to__m_10/Relu?
/L-14-m_0-1/m_10__to__m_11/MatMul/ReadVariableOpReadVariableOp8l_14_m_0_1_m_10__to__m_11_matmul_readvariableop_resource*
_output_shapes

:11*
dtype021
/L-14-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp?
 L-14-m_0-1/m_10__to__m_11/MatMulMatMul+L-14-m_0-1/m_9__to__m_10/Relu:activations:07L-14-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12"
 L-14-m_0-1/m_10__to__m_11/MatMul?
0L-14-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOpReadVariableOp9l_14_m_0_1_m_10__to__m_11_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype022
0L-14-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp?
!L-14-m_0-1/m_10__to__m_11/BiasAddAdd*L-14-m_0-1/m_10__to__m_11/MatMul:product:08L-14-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12#
!L-14-m_0-1/m_10__to__m_11/BiasAdd?
L-14-m_0-1/m_10__to__m_11/ReluRelu%L-14-m_0-1/m_10__to__m_11/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_10__to__m_11/Relu?
/L-14-m_0-1/m_11__to__m_12/MatMul/ReadVariableOpReadVariableOp8l_14_m_0_1_m_11__to__m_12_matmul_readvariableop_resource*
_output_shapes

:11*
dtype021
/L-14-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp?
 L-14-m_0-1/m_11__to__m_12/MatMulMatMul,L-14-m_0-1/m_10__to__m_11/Relu:activations:07L-14-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12"
 L-14-m_0-1/m_11__to__m_12/MatMul?
0L-14-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOpReadVariableOp9l_14_m_0_1_m_11__to__m_12_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype022
0L-14-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp?
!L-14-m_0-1/m_11__to__m_12/BiasAddAdd*L-14-m_0-1/m_11__to__m_12/MatMul:product:08L-14-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12#
!L-14-m_0-1/m_11__to__m_12/BiasAdd?
L-14-m_0-1/m_11__to__m_12/ReluRelu%L-14-m_0-1/m_11__to__m_12/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_11__to__m_12/Relu?
/L-14-m_0-1/m_12__to__m_13/MatMul/ReadVariableOpReadVariableOp8l_14_m_0_1_m_12__to__m_13_matmul_readvariableop_resource*
_output_shapes

:11*
dtype021
/L-14-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp?
 L-14-m_0-1/m_12__to__m_13/MatMulMatMul,L-14-m_0-1/m_11__to__m_12/Relu:activations:07L-14-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12"
 L-14-m_0-1/m_12__to__m_13/MatMul?
0L-14-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOpReadVariableOp9l_14_m_0_1_m_12__to__m_13_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype022
0L-14-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp?
!L-14-m_0-1/m_12__to__m_13/BiasAddAdd*L-14-m_0-1/m_12__to__m_13/MatMul:product:08L-14-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12#
!L-14-m_0-1/m_12__to__m_13/BiasAdd?
L-14-m_0-1/m_12__to__m_13/ReluRelu%L-14-m_0-1/m_12__to__m_13/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_12__to__m_13/Relu?
/L-14-m_0-1/m_13__to__m_14/MatMul/ReadVariableOpReadVariableOp8l_14_m_0_1_m_13__to__m_14_matmul_readvariableop_resource*
_output_shapes

:11*
dtype021
/L-14-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp?
 L-14-m_0-1/m_13__to__m_14/MatMulMatMul,L-14-m_0-1/m_12__to__m_13/Relu:activations:07L-14-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12"
 L-14-m_0-1/m_13__to__m_14/MatMul?
0L-14-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOpReadVariableOp9l_14_m_0_1_m_13__to__m_14_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype022
0L-14-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp?
!L-14-m_0-1/m_13__to__m_14/BiasAddAdd*L-14-m_0-1/m_13__to__m_14/MatMul:product:08L-14-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12#
!L-14-m_0-1/m_13__to__m_14/BiasAdd?
L-14-m_0-1/m_13__to__m_14/ReluRelu%L-14-m_0-1/m_13__to__m_14/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-1/m_13__to__m_14/Relu?
/L-14-m_0-1/m_14__to__m_15/MatMul/ReadVariableOpReadVariableOp8l_14_m_0_1_m_14__to__m_15_matmul_readvariableop_resource*
_output_shapes

:1*
dtype021
/L-14-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp?
 L-14-m_0-1/m_14__to__m_15/MatMulMatMul,L-14-m_0-1/m_13__to__m_14/Relu:activations:07L-14-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 L-14-m_0-1/m_14__to__m_15/MatMul?
0L-14-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOpReadVariableOp9l_14_m_0_1_m_14__to__m_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0L-14-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp?
!L-14-m_0-1/m_14__to__m_15/BiasAddAdd*L-14-m_0-1/m_14__to__m_15/MatMul:product:08L-14-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!L-14-m_0-1/m_14__to__m_15/BiasAdd?
!L-14-m_0-1/m_14__to__m_15/SoftmaxSoftmax%L-14-m_0-1/m_14__to__m_15/BiasAdd:z:0*
T0*'
_output_shapes
:?????????2#
!L-14-m_0-1/m_14__to__m_15/Softmax?
IdentityIdentity+L-14-m_0-1/m_14__to__m_15/Softmax:softmax:0/^L-14-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp.^L-14-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp1^L-14-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp0^L-14-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp1^L-14-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp0^L-14-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp1^L-14-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp0^L-14-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp1^L-14-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp0^L-14-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp1^L-14-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp0^L-14-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp/^L-14-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp.^L-14-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp/^L-14-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp.^L-14-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp/^L-14-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp.^L-14-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp/^L-14-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp.^L-14-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp/^L-14-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp.^L-14-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp/^L-14-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp.^L-14-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp/^L-14-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp.^L-14-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp/^L-14-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp.^L-14-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp0^L-14-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp/^L-14-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.L-14-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp.L-14-m_0-1/m_0__to__m_1/BiasAdd/ReadVariableOp2^
-L-14-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp-L-14-m_0-1/m_0__to__m_1/MatMul/ReadVariableOp2d
0L-14-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp0L-14-m_0-1/m_10__to__m_11/BiasAdd/ReadVariableOp2b
/L-14-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp/L-14-m_0-1/m_10__to__m_11/MatMul/ReadVariableOp2d
0L-14-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp0L-14-m_0-1/m_11__to__m_12/BiasAdd/ReadVariableOp2b
/L-14-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp/L-14-m_0-1/m_11__to__m_12/MatMul/ReadVariableOp2d
0L-14-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp0L-14-m_0-1/m_12__to__m_13/BiasAdd/ReadVariableOp2b
/L-14-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp/L-14-m_0-1/m_12__to__m_13/MatMul/ReadVariableOp2d
0L-14-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp0L-14-m_0-1/m_13__to__m_14/BiasAdd/ReadVariableOp2b
/L-14-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp/L-14-m_0-1/m_13__to__m_14/MatMul/ReadVariableOp2d
0L-14-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp0L-14-m_0-1/m_14__to__m_15/BiasAdd/ReadVariableOp2b
/L-14-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp/L-14-m_0-1/m_14__to__m_15/MatMul/ReadVariableOp2`
.L-14-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp.L-14-m_0-1/m_1__to__m_2/BiasAdd/ReadVariableOp2^
-L-14-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp-L-14-m_0-1/m_1__to__m_2/MatMul/ReadVariableOp2`
.L-14-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp.L-14-m_0-1/m_2__to__m_3/BiasAdd/ReadVariableOp2^
-L-14-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp-L-14-m_0-1/m_2__to__m_3/MatMul/ReadVariableOp2`
.L-14-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp.L-14-m_0-1/m_3__to__m_4/BiasAdd/ReadVariableOp2^
-L-14-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp-L-14-m_0-1/m_3__to__m_4/MatMul/ReadVariableOp2`
.L-14-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp.L-14-m_0-1/m_4__to__m_5/BiasAdd/ReadVariableOp2^
-L-14-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp-L-14-m_0-1/m_4__to__m_5/MatMul/ReadVariableOp2`
.L-14-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp.L-14-m_0-1/m_5__to__m_6/BiasAdd/ReadVariableOp2^
-L-14-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp-L-14-m_0-1/m_5__to__m_6/MatMul/ReadVariableOp2`
.L-14-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp.L-14-m_0-1/m_6__to__m_7/BiasAdd/ReadVariableOp2^
-L-14-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp-L-14-m_0-1/m_6__to__m_7/MatMul/ReadVariableOp2`
.L-14-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp.L-14-m_0-1/m_7__to__m_8/BiasAdd/ReadVariableOp2^
-L-14-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp-L-14-m_0-1/m_7__to__m_8/MatMul/ReadVariableOp2`
.L-14-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp.L-14-m_0-1/m_8__to__m_9/BiasAdd/ReadVariableOp2^
-L-14-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp-L-14-m_0-1/m_8__to__m_9/MatMul/ReadVariableOp2b
/L-14-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp/L-14-m_0-1/m_9__to__m_10/BiasAdd/ReadVariableOp2`
.L-14-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp.L-14-m_0-1/m_9__to__m_10/MatMul/ReadVariableOp:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_23584002

inputs0
matmul_readvariableop_resource:11-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:1*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12	
BiasAddS
ReluReluBiasAdd:z:0*
T0*'
_output_shapes
:?????????12
Relu?
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:11*
dtype026
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?
%m_13__to__m_14/kernel/Regularizer/AbsAbs<m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:112'
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
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2l
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????1
 
_user_specified_nameinputs
?
?
/__inference_m_3__to__m_4_layer_call_fn_23585856

inputs
unknown:11
	unknown_0:1
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_235837722
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????1: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????1
 
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
m_14__to__m_150
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??

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
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
?_default_save_signature
+?&call_and_return_all_conditional_losses
?__call__"??
_tf_keras_sequential??{"name": "L-14-m_0-1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "L-14-m_0-1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "m_0__to__m_1_input"}}, {"class_name": "Dense", "config": {"name": "m_0__to__m_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_1__to__m_2", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_2__to__m_3", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_3__to__m_4", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_4__to__m_5", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_5__to__m_6", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_6__to__m_7", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_7__to__m_8", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_8__to__m_9", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_9__to__m_10", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_10__to__m_11", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_11__to__m_12", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_12__to__m_13", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_13__to__m_14", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_14__to__m_15", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 61, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "m_0__to__m_1_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "L-14-m_0-1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "m_0__to__m_1_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "m_0__to__m_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}, {"class_name": "Dense", "config": {"name": "m_1__to__m_2", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 7}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}, {"class_name": "Dense", "config": {"name": "m_2__to__m_3", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "m_3__to__m_4", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 15}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "Dense", "config": {"name": "m_4__to__m_5", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 19}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "Dense", "config": {"name": "m_5__to__m_6", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 23}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "Dense", "config": {"name": "m_6__to__m_7", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 27}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "m_7__to__m_8", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dense", "config": {"name": "m_8__to__m_9", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 35}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36}, {"class_name": "Dense", "config": {"name": "m_9__to__m_10", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 39}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40}, {"class_name": "Dense", "config": {"name": "m_10__to__m_11", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 43}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 44}, {"class_name": "Dense", "config": {"name": "m_11__to__m_12", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 47}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 48}, {"class_name": "Dense", "config": {"name": "m_12__to__m_13", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 51}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 52}, {"class_name": "Dense", "config": {"name": "m_13__to__m_14", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 55}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 56}, {"class_name": "Dense", "config": {"name": "m_14__to__m_15", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 59}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0005000000237487257, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_0__to__m_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_0__to__m_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}
?	

kernel
bias
regularization_losses
	variables
 trainable_variables
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_1__to__m_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_1__to__m_2", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 7}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 63}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

"kernel
#bias
$regularization_losses
%	variables
&trainable_variables
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_2__to__m_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_2__to__m_3", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

(kernel
)bias
*regularization_losses
+	variables
,trainable_variables
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_3__to__m_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_3__to__m_4", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 15}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

.kernel
/bias
0regularization_losses
1	variables
2trainable_variables
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_4__to__m_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_4__to__m_5", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 19}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

4kernel
5bias
6regularization_losses
7	variables
8trainable_variables
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_5__to__m_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_5__to__m_6", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 23}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

:kernel
;bias
<regularization_losses
=	variables
>trainable_variables
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_6__to__m_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_6__to__m_7", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 27}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

@kernel
Abias
Bregularization_losses
C	variables
Dtrainable_variables
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_7__to__m_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_7__to__m_8", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

Fkernel
Gbias
Hregularization_losses
I	variables
Jtrainable_variables
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_8__to__m_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_8__to__m_9", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 35}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

Lkernel
Mbias
Nregularization_losses
O	variables
Ptrainable_variables
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_9__to__m_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_9__to__m_10", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 39}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

Rkernel
Sbias
Tregularization_losses
U	variables
Vtrainable_variables
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_10__to__m_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_10__to__m_11", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 43}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

Xkernel
Ybias
Zregularization_losses
[	variables
\trainable_variables
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_11__to__m_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_11__to__m_12", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 47}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

^kernel
_bias
`regularization_losses
a	variables
btrainable_variables
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_12__to__m_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_12__to__m_13", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 51}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 74}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

dkernel
ebias
fregularization_losses
g	variables
htrainable_variables
i	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_13__to__m_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_13__to__m_14", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 55}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 56, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

jkernel
kbias
lregularization_losses
m	variables
ntrainable_variables
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_14__to__m_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_14__to__m_15", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 59}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 76}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?
piter

qbeta_1

rbeta_2
	sdecay
tlearning_ratem?m?m?m?"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?@m?Am?Fm?Gm?Lm?Mm?Rm?Sm?Xm?Ym?^m?_m?dm?em?jm?km?v?v?v?v?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?@v?Av?Fv?Gv?Lv?Mv?Rv?Sv?Xv?Yv?^v?_v?dv?ev?jv?kv?"
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
?14"
trackable_list_wrapper
?
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511
:12
;13
@14
A15
F16
G17
L18
M19
R20
S21
X22
Y23
^24
_25
d26
e27
j28
k29"
trackable_list_wrapper
?
0
1
2
3
"4
#5
(6
)7
.8
/9
410
511
:12
;13
@14
A15
F16
G17
L18
M19
R20
S21
X22
Y23
^24
_25
d26
e27
j28
k29"
trackable_list_wrapper
?
regularization_losses
ulayer_regularization_losses
vmetrics
wnon_trainable_variables

xlayers
	variables
trainable_variables
ylayer_metrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:#12m_0__to__m_1/kernel
:12m_0__to__m_1/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
zlayer_regularization_losses
{metrics
|non_trainable_variables

}layers
	variables
trainable_variables
~layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_1__to__m_2/kernel
:12m_1__to__m_2/bias
(
?0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
regularization_losses
layer_regularization_losses
?metrics
?non_trainable_variables
?layers
	variables
 trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_2__to__m_3/kernel
:12m_2__to__m_3/bias
(
?0"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?
$regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
%	variables
&trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_3__to__m_4/kernel
:12m_3__to__m_4/bias
(
?0"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
*regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
+	variables
,trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_4__to__m_5/kernel
:12m_4__to__m_5/bias
(
?0"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
0regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
1	variables
2trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_5__to__m_6/kernel
:12m_5__to__m_6/bias
(
?0"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
?
6regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
7	variables
8trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_6__to__m_7/kernel
:12m_6__to__m_7/bias
(
?0"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
?
<regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
=	variables
>trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_7__to__m_8/kernel
:12m_7__to__m_8/bias
(
?0"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
?
Bregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
C	variables
Dtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_8__to__m_9/kernel
:12m_8__to__m_9/bias
(
?0"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
?
Hregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
I	variables
Jtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$112m_9__to__m_10/kernel
 :12m_9__to__m_10/bias
(
?0"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
?
Nregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
O	variables
Ptrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%112m_10__to__m_11/kernel
!:12m_10__to__m_11/bias
(
?0"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
?
Tregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
U	variables
Vtrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%112m_11__to__m_12/kernel
!:12m_11__to__m_12/bias
(
?0"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
?
Zregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
[	variables
\trainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%112m_12__to__m_13/kernel
!:12m_12__to__m_13/bias
(
?0"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
?
`regularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
a	variables
btrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%112m_13__to__m_14/kernel
!:12m_13__to__m_14/bias
(
?0"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
?
fregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
g	variables
htrainable_variables
?layer_metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%12m_14__to__m_15/kernel
!:2m_14__to__m_15/bias
(
?0"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
?
lregularization_losses
 ?layer_regularization_losses
?metrics
?non_trainable_variables
?layers
m	variables
ntrainable_variables
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
14"
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
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 77}
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
*:(12Adam/m_0__to__m_1/kernel/m
$:"12Adam/m_0__to__m_1/bias/m
*:(112Adam/m_1__to__m_2/kernel/m
$:"12Adam/m_1__to__m_2/bias/m
*:(112Adam/m_2__to__m_3/kernel/m
$:"12Adam/m_2__to__m_3/bias/m
*:(112Adam/m_3__to__m_4/kernel/m
$:"12Adam/m_3__to__m_4/bias/m
*:(112Adam/m_4__to__m_5/kernel/m
$:"12Adam/m_4__to__m_5/bias/m
*:(112Adam/m_5__to__m_6/kernel/m
$:"12Adam/m_5__to__m_6/bias/m
*:(112Adam/m_6__to__m_7/kernel/m
$:"12Adam/m_6__to__m_7/bias/m
*:(112Adam/m_7__to__m_8/kernel/m
$:"12Adam/m_7__to__m_8/bias/m
*:(112Adam/m_8__to__m_9/kernel/m
$:"12Adam/m_8__to__m_9/bias/m
+:)112Adam/m_9__to__m_10/kernel/m
%:#12Adam/m_9__to__m_10/bias/m
,:*112Adam/m_10__to__m_11/kernel/m
&:$12Adam/m_10__to__m_11/bias/m
,:*112Adam/m_11__to__m_12/kernel/m
&:$12Adam/m_11__to__m_12/bias/m
,:*112Adam/m_12__to__m_13/kernel/m
&:$12Adam/m_12__to__m_13/bias/m
,:*112Adam/m_13__to__m_14/kernel/m
&:$12Adam/m_13__to__m_14/bias/m
,:*12Adam/m_14__to__m_15/kernel/m
&:$2Adam/m_14__to__m_15/bias/m
*:(12Adam/m_0__to__m_1/kernel/v
$:"12Adam/m_0__to__m_1/bias/v
*:(112Adam/m_1__to__m_2/kernel/v
$:"12Adam/m_1__to__m_2/bias/v
*:(112Adam/m_2__to__m_3/kernel/v
$:"12Adam/m_2__to__m_3/bias/v
*:(112Adam/m_3__to__m_4/kernel/v
$:"12Adam/m_3__to__m_4/bias/v
*:(112Adam/m_4__to__m_5/kernel/v
$:"12Adam/m_4__to__m_5/bias/v
*:(112Adam/m_5__to__m_6/kernel/v
$:"12Adam/m_5__to__m_6/bias/v
*:(112Adam/m_6__to__m_7/kernel/v
$:"12Adam/m_6__to__m_7/bias/v
*:(112Adam/m_7__to__m_8/kernel/v
$:"12Adam/m_7__to__m_8/bias/v
*:(112Adam/m_8__to__m_9/kernel/v
$:"12Adam/m_8__to__m_9/bias/v
+:)112Adam/m_9__to__m_10/kernel/v
%:#12Adam/m_9__to__m_10/bias/v
,:*112Adam/m_10__to__m_11/kernel/v
&:$12Adam/m_10__to__m_11/bias/v
,:*112Adam/m_11__to__m_12/kernel/v
&:$12Adam/m_11__to__m_12/bias/v
,:*112Adam/m_12__to__m_13/kernel/v
&:$12Adam/m_12__to__m_13/bias/v
,:*112Adam/m_13__to__m_14/kernel/v
&:$12Adam/m_13__to__m_14/bias/v
,:*12Adam/m_14__to__m_15/kernel/v
&:$2Adam/m_14__to__m_15/bias/v
?2?
#__inference__wrapped_model_23583679?
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
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23585399
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23585598
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23584868
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23585037?
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
-__inference_L-14-m_0-1_layer_call_fn_23584185
-__inference_L-14-m_0-1_layer_call_fn_23585663
-__inference_L-14-m_0-1_layer_call_fn_23585728
-__inference_L-14-m_0-1_layer_call_fn_23584699?
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
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_23585751?
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
/__inference_m_0__to__m_1_layer_call_fn_23585760?
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
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_23585783?
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
/__inference_m_1__to__m_2_layer_call_fn_23585792?
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
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_23585815?
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
/__inference_m_2__to__m_3_layer_call_fn_23585824?
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
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_23585847?
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
/__inference_m_3__to__m_4_layer_call_fn_23585856?
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
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_23585879?
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
/__inference_m_4__to__m_5_layer_call_fn_23585888?
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
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_23585911?
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
/__inference_m_5__to__m_6_layer_call_fn_23585920?
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
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_23585943?
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
/__inference_m_6__to__m_7_layer_call_fn_23585952?
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
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_23585975?
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
/__inference_m_7__to__m_8_layer_call_fn_23585984?
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
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_23586007?
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
/__inference_m_8__to__m_9_layer_call_fn_23586016?
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
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_23586039?
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
0__inference_m_9__to__m_10_layer_call_fn_23586048?
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
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_23586071?
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
1__inference_m_10__to__m_11_layer_call_fn_23586080?
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
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_23586103?
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
1__inference_m_11__to__m_12_layer_call_fn_23586112?
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
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_23586135?
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
1__inference_m_12__to__m_13_layer_call_fn_23586144?
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
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_23586167?
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
1__inference_m_13__to__m_14_layer_call_fn_23586176?
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
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_23586199?
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
1__inference_m_14__to__m_15_layer_call_fn_23586208?
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
__inference_loss_fn_0_23586219?
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
__inference_loss_fn_1_23586230?
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
__inference_loss_fn_2_23586241?
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
__inference_loss_fn_3_23586252?
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
__inference_loss_fn_4_23586263?
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
__inference_loss_fn_5_23586274?
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
__inference_loss_fn_6_23586285?
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
__inference_loss_fn_7_23586296?
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
__inference_loss_fn_8_23586307?
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
__inference_loss_fn_9_23586318?
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
__inference_loss_fn_10_23586329?
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
__inference_loss_fn_11_23586340?
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
__inference_loss_fn_12_23586351?
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
__inference_loss_fn_13_23586362?
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
__inference_loss_fn_14_23586373?
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
&__inference_signature_wrapper_23585200m_0__to__m_1_input"?
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
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23584868?"#()./45:;@AFGLMRSXY^_dejkC?@
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
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23585037?"#()./45:;@AFGLMRSXY^_dejkC?@
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
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23585399?"#()./45:;@AFGLMRSXY^_dejk7?4
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
H__inference_L-14-m_0-1_layer_call_and_return_conditional_losses_23585598?"#()./45:;@AFGLMRSXY^_dejk7?4
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
-__inference_L-14-m_0-1_layer_call_fn_23584185"#()./45:;@AFGLMRSXY^_dejkC?@
9?6
,?)
m_0__to__m_1_input?????????
p 

 
? "???????????
-__inference_L-14-m_0-1_layer_call_fn_23584699"#()./45:;@AFGLMRSXY^_dejkC?@
9?6
,?)
m_0__to__m_1_input?????????
p

 
? "???????????
-__inference_L-14-m_0-1_layer_call_fn_23585663s"#()./45:;@AFGLMRSXY^_dejk7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
-__inference_L-14-m_0-1_layer_call_fn_23585728s"#()./45:;@AFGLMRSXY^_dejk7?4
-?*
 ?
inputs?????????
p

 
? "???????????
#__inference__wrapped_model_23583679?"#()./45:;@AFGLMRSXY^_dejk;?8
1?.
,?)
m_0__to__m_1_input?????????
? "??<
:
m_14__to__m_15(?%
m_14__to__m_15?????????=
__inference_loss_fn_0_23586219?

? 
? "? >
__inference_loss_fn_10_23586329R?

? 
? "? >
__inference_loss_fn_11_23586340X?

? 
? "? >
__inference_loss_fn_12_23586351^?

? 
? "? >
__inference_loss_fn_13_23586362d?

? 
? "? >
__inference_loss_fn_14_23586373j?

? 
? "? =
__inference_loss_fn_1_23586230?

? 
? "? =
__inference_loss_fn_2_23586241"?

? 
? "? =
__inference_loss_fn_3_23586252(?

? 
? "? =
__inference_loss_fn_4_23586263.?

? 
? "? =
__inference_loss_fn_5_235862744?

? 
? "? =
__inference_loss_fn_6_23586285:?

? 
? "? =
__inference_loss_fn_7_23586296@?

? 
? "? =
__inference_loss_fn_8_23586307F?

? 
? "? =
__inference_loss_fn_9_23586318L?

? 
? "? ?
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_23585751\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????1
? ?
/__inference_m_0__to__m_1_layer_call_fn_23585760O/?,
%?"
 ?
inputs?????????
? "??????????1?
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_23586071\RS/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
1__inference_m_10__to__m_11_layer_call_fn_23586080ORS/?,
%?"
 ?
inputs?????????1
? "??????????1?
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_23586103\XY/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
1__inference_m_11__to__m_12_layer_call_fn_23586112OXY/?,
%?"
 ?
inputs?????????1
? "??????????1?
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_23586135\^_/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
1__inference_m_12__to__m_13_layer_call_fn_23586144O^_/?,
%?"
 ?
inputs?????????1
? "??????????1?
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_23586167\de/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
1__inference_m_13__to__m_14_layer_call_fn_23586176Ode/?,
%?"
 ?
inputs?????????1
? "??????????1?
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_23586199\jk/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????
? ?
1__inference_m_14__to__m_15_layer_call_fn_23586208Ojk/?,
%?"
 ?
inputs?????????1
? "???????????
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_23585783\/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_1__to__m_2_layer_call_fn_23585792O/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_23585815\"#/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_2__to__m_3_layer_call_fn_23585824O"#/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_23585847\()/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_3__to__m_4_layer_call_fn_23585856O()/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_23585879\.//?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_4__to__m_5_layer_call_fn_23585888O.//?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_23585911\45/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_5__to__m_6_layer_call_fn_23585920O45/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_23585943\:;/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_6__to__m_7_layer_call_fn_23585952O:;/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_23585975\@A/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_7__to__m_8_layer_call_fn_23585984O@A/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_23586007\FG/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_8__to__m_9_layer_call_fn_23586016OFG/?,
%?"
 ?
inputs?????????1
? "??????????1?
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_23586039\LM/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
0__inference_m_9__to__m_10_layer_call_fn_23586048OLM/?,
%?"
 ?
inputs?????????1
? "??????????1?
&__inference_signature_wrapper_23585200?"#()./45:;@AFGLMRSXY^_dejkQ?N
? 
G?D
B
m_0__to__m_1_input,?)
m_0__to__m_1_input?????????"??<
:
m_14__to__m_15(?%
m_14__to__m_15?????????