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
:1*$
shared_namem_0__to__m_1/kernel
{
'm_0__to__m_1/kernel/Read/ReadVariableOpReadVariableOpm_0__to__m_1/kernel*
_output_shapes

:1*
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
:1*&
shared_namem_14__to__m_15/kernel

)m_14__to__m_15/kernel/Read/ReadVariableOpReadVariableOpm_14__to__m_15/kernel*
_output_shapes

:1*
dtype0
~
m_14__to__m_15/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_namem_14__to__m_15/bias
w
'm_14__to__m_15/bias/Read/ReadVariableOpReadVariableOpm_14__to__m_15/bias*
_output_shapes
:*
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
:1*+
shared_nameAdam/m_0__to__m_1/kernel/m
?
.Adam/m_0__to__m_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_0__to__m_1/kernel/m*
_output_shapes

:1*
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
:1*-
shared_nameAdam/m_14__to__m_15/kernel/m
?
0Adam/m_14__to__m_15/kernel/m/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/kernel/m*
_output_shapes

:1*
dtype0
?
Adam/m_14__to__m_15/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m_14__to__m_15/bias/m
?
.Adam/m_14__to__m_15/bias/m/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/bias/m*
_output_shapes
:*
dtype0
?
Adam/m_0__to__m_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:1*+
shared_nameAdam/m_0__to__m_1/kernel/v
?
.Adam/m_0__to__m_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_0__to__m_1/kernel/v*
_output_shapes

:1*
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
:1*-
shared_nameAdam/m_14__to__m_15/kernel/v
?
0Adam/m_14__to__m_15/kernel/v/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/kernel/v*
_output_shapes

:1*
dtype0
?
Adam/m_14__to__m_15/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*+
shared_nameAdam/m_14__to__m_15/bias/v
?
.Adam/m_14__to__m_15/bias/v/Read/ReadVariableOpReadVariableOpAdam/m_14__to__m_15/bias/v*
_output_shapes
:*
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
h

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
h

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
h

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
h

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
h

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
h

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
h

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
h

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
h

^kernel
_bias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
h

dkernel
ebias
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
h

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
?
piter

qbeta_1

rbeta_2
	sdecay
tlearning_ratem?m?m?m?"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?@m?Am?Fm?Gm?Lm?Mm?Rm?Sm?Xm?Ym?^m?_m?dm?em?jm?km?v?v?v?v?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?@v?Av?Fv?Gv?Lv?Mv?Rv?Sv?Xv?Yv?^v?_v?dv?ev?jv?kv?
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
 
?
ulayer_metrics
	variables
trainable_variables
vlayer_regularization_losses
regularization_losses

wlayers
xnon_trainable_variables
ymetrics
 
_]
VARIABLE_VALUEm_0__to__m_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_0__to__m_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
zlayer_metrics
	variables
trainable_variables
{layer_regularization_losses
regularization_losses

|layers
}non_trainable_variables
~metrics
_]
VARIABLE_VALUEm_1__to__m_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_1__to__m_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
layer_metrics
	variables
trainable_variables
 ?layer_regularization_losses
 regularization_losses
?layers
?non_trainable_variables
?metrics
_]
VARIABLE_VALUEm_2__to__m_3/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_2__to__m_3/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

"0
#1

"0
#1
 
?
?layer_metrics
$	variables
%trainable_variables
 ?layer_regularization_losses
&regularization_losses
?layers
?non_trainable_variables
?metrics
_]
VARIABLE_VALUEm_3__to__m_4/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_3__to__m_4/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

(0
)1

(0
)1
 
?
?layer_metrics
*	variables
+trainable_variables
 ?layer_regularization_losses
,regularization_losses
?layers
?non_trainable_variables
?metrics
_]
VARIABLE_VALUEm_4__to__m_5/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_4__to__m_5/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
?
?layer_metrics
0	variables
1trainable_variables
 ?layer_regularization_losses
2regularization_losses
?layers
?non_trainable_variables
?metrics
_]
VARIABLE_VALUEm_5__to__m_6/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_5__to__m_6/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
?
?layer_metrics
6	variables
7trainable_variables
 ?layer_regularization_losses
8regularization_losses
?layers
?non_trainable_variables
?metrics
_]
VARIABLE_VALUEm_6__to__m_7/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_6__to__m_7/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

:0
;1

:0
;1
 
?
?layer_metrics
<	variables
=trainable_variables
 ?layer_regularization_losses
>regularization_losses
?layers
?non_trainable_variables
?metrics
_]
VARIABLE_VALUEm_7__to__m_8/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_7__to__m_8/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

@0
A1

@0
A1
 
?
?layer_metrics
B	variables
Ctrainable_variables
 ?layer_regularization_losses
Dregularization_losses
?layers
?non_trainable_variables
?metrics
_]
VARIABLE_VALUEm_8__to__m_9/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
[Y
VARIABLE_VALUEm_8__to__m_9/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

F0
G1

F0
G1
 
?
?layer_metrics
H	variables
Itrainable_variables
 ?layer_regularization_losses
Jregularization_losses
?layers
?non_trainable_variables
?metrics
`^
VARIABLE_VALUEm_9__to__m_10/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEm_9__to__m_10/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

L0
M1

L0
M1
 
?
?layer_metrics
N	variables
Otrainable_variables
 ?layer_regularization_losses
Pregularization_losses
?layers
?non_trainable_variables
?metrics
b`
VARIABLE_VALUEm_10__to__m_11/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_10__to__m_11/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

R0
S1

R0
S1
 
?
?layer_metrics
T	variables
Utrainable_variables
 ?layer_regularization_losses
Vregularization_losses
?layers
?non_trainable_variables
?metrics
b`
VARIABLE_VALUEm_11__to__m_12/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_11__to__m_12/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

X0
Y1

X0
Y1
 
?
?layer_metrics
Z	variables
[trainable_variables
 ?layer_regularization_losses
\regularization_losses
?layers
?non_trainable_variables
?metrics
b`
VARIABLE_VALUEm_12__to__m_13/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_12__to__m_13/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

^0
_1

^0
_1
 
?
?layer_metrics
`	variables
atrainable_variables
 ?layer_regularization_losses
bregularization_losses
?layers
?non_trainable_variables
?metrics
b`
VARIABLE_VALUEm_13__to__m_14/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_13__to__m_14/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

d0
e1

d0
e1
 
?
?layer_metrics
f	variables
gtrainable_variables
 ?layer_regularization_losses
hregularization_losses
?layers
?non_trainable_variables
?metrics
b`
VARIABLE_VALUEm_14__to__m_15/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
^\
VARIABLE_VALUEm_14__to__m_15/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

j0
k1

j0
k1
 
?
?layer_metrics
l	variables
mtrainable_variables
 ?layer_regularization_losses
nregularization_losses
?layers
?non_trainable_variables
?metrics
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

?0
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
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCall"serving_default_m_0__to__m_1_inputm_0__to__m_1/kernelm_0__to__m_1/biasm_1__to__m_2/kernelm_1__to__m_2/biasm_2__to__m_3/kernelm_2__to__m_3/biasm_3__to__m_4/kernelm_3__to__m_4/biasm_4__to__m_5/kernelm_4__to__m_5/biasm_5__to__m_6/kernelm_5__to__m_6/biasm_6__to__m_7/kernelm_6__to__m_7/biasm_7__to__m_8/kernelm_7__to__m_8/biasm_8__to__m_9/kernelm_8__to__m_9/biasm_9__to__m_10/kernelm_9__to__m_10/biasm_10__to__m_11/kernelm_10__to__m_11/biasm_11__to__m_12/kernelm_11__to__m_12/biasm_12__to__m_13/kernelm_12__to__m_13/biasm_13__to__m_14/kernelm_13__to__m_14/biasm_14__to__m_15/kernelm_14__to__m_15/bias**
Tin#
!2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? */
f*R(
&__inference_signature_wrapper_14384786
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
!__inference__traced_save_14386273
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
$__inference__traced_restore_14386574??
?
?
__inference_loss_fn_11_14385926O
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
?
?
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_14383519

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
__inference_loss_fn_2_14385827M
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
?
?
/__inference_m_5__to__m_6_layer_call_fn_14385506

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
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_143834042
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
1__inference_m_12__to__m_13_layer_call_fn_14385730

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
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_143835652
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
/__inference_m_3__to__m_4_layer_call_fn_14385442

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
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_143833582
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
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_14383358

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
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14384985

inputs=
+m_0__to__m_1_matmul_readvariableop_resource:1:
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
-m_14__to__m_15_matmul_readvariableop_resource:1<
.m_14__to__m_15_biasadd_readvariableop_resource:
identity??#m_0__to__m_1/BiasAdd/ReadVariableOp?"m_0__to__m_1/MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?%m_10__to__m_11/BiasAdd/ReadVariableOp?$m_10__to__m_11/MatMul/ReadVariableOp?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?%m_11__to__m_12/BiasAdd/ReadVariableOp?$m_11__to__m_12/MatMul/ReadVariableOp?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?%m_12__to__m_13/BiasAdd/ReadVariableOp?$m_12__to__m_13/MatMul/ReadVariableOp?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?%m_13__to__m_14/BiasAdd/ReadVariableOp?$m_13__to__m_14/MatMul/ReadVariableOp?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?%m_14__to__m_15/BiasAdd/ReadVariableOp?$m_14__to__m_15/MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?#m_1__to__m_2/BiasAdd/ReadVariableOp?"m_1__to__m_2/MatMul/ReadVariableOp?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?#m_2__to__m_3/BiasAdd/ReadVariableOp?"m_2__to__m_3/MatMul/ReadVariableOp?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?#m_3__to__m_4/BiasAdd/ReadVariableOp?"m_3__to__m_4/MatMul/ReadVariableOp?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?#m_4__to__m_5/BiasAdd/ReadVariableOp?"m_4__to__m_5/MatMul/ReadVariableOp?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?#m_5__to__m_6/BiasAdd/ReadVariableOp?"m_5__to__m_6/MatMul/ReadVariableOp?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?#m_6__to__m_7/BiasAdd/ReadVariableOp?"m_6__to__m_7/MatMul/ReadVariableOp?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?#m_7__to__m_8/BiasAdd/ReadVariableOp?"m_7__to__m_8/MatMul/ReadVariableOp?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?#m_8__to__m_9/BiasAdd/ReadVariableOp?"m_8__to__m_9/MatMul/ReadVariableOp?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?$m_9__to__m_10/BiasAdd/ReadVariableOp?#m_9__to__m_10/MatMul/ReadVariableOp?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
"m_0__to__m_1/MatMul/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:1*
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

:1*
dtype02&
$m_14__to__m_15/MatMul/ReadVariableOp?
m_14__to__m_15/MatMulMatMul!m_13__to__m_14/Relu:activations:0,m_14__to__m_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/MatMul?
%m_14__to__m_15/BiasAdd/ReadVariableOpReadVariableOp.m_14__to__m_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%m_14__to__m_15/BiasAdd/ReadVariableOp?
m_14__to__m_15/BiasAddAddm_14__to__m_15/MatMul:product:0-m_14__to__m_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/BiasAdd?
m_14__to__m_15/SoftmaxSoftmaxm_14__to__m_15/BiasAdd:z:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/Softmax?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
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
:?????????
 
_user_specified_nameinputs
?
?
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_14385689

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
?
?
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_14383542

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
?
?
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_14385657

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
?
?
/__inference_m_8__to__m_9_layer_call_fn_14385602

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
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_143834732
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
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_14383496

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
?
?
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_14383335

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
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_14383381

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
?
?
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_14383588

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
1__inference_m_13__to__m_14_layer_call_fn_14385762

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
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_143835882
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
__inference_loss_fn_0_14385805M
;m_0__to__m_1_kernel_regularizer_abs_readvariableop_resource:1
identity??2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp;m_0__to__m_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
?
?
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_14385337

inputs0
matmul_readvariableop_resource:1-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
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

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_14385401

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
?
?
__inference_loss_fn_3_14385838M
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
?
?
__inference_loss_fn_5_14385860M
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
?
?
1__inference_m_10__to__m_11_layer_call_fn_14385666

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
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_143835192
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
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_14385369

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
??
?
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14385184

inputs=
+m_0__to__m_1_matmul_readvariableop_resource:1:
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
-m_14__to__m_15_matmul_readvariableop_resource:1<
.m_14__to__m_15_biasadd_readvariableop_resource:
identity??#m_0__to__m_1/BiasAdd/ReadVariableOp?"m_0__to__m_1/MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?%m_10__to__m_11/BiasAdd/ReadVariableOp?$m_10__to__m_11/MatMul/ReadVariableOp?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?%m_11__to__m_12/BiasAdd/ReadVariableOp?$m_11__to__m_12/MatMul/ReadVariableOp?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?%m_12__to__m_13/BiasAdd/ReadVariableOp?$m_12__to__m_13/MatMul/ReadVariableOp?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?%m_13__to__m_14/BiasAdd/ReadVariableOp?$m_13__to__m_14/MatMul/ReadVariableOp?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?%m_14__to__m_15/BiasAdd/ReadVariableOp?$m_14__to__m_15/MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?#m_1__to__m_2/BiasAdd/ReadVariableOp?"m_1__to__m_2/MatMul/ReadVariableOp?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?#m_2__to__m_3/BiasAdd/ReadVariableOp?"m_2__to__m_3/MatMul/ReadVariableOp?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?#m_3__to__m_4/BiasAdd/ReadVariableOp?"m_3__to__m_4/MatMul/ReadVariableOp?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?#m_4__to__m_5/BiasAdd/ReadVariableOp?"m_4__to__m_5/MatMul/ReadVariableOp?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?#m_5__to__m_6/BiasAdd/ReadVariableOp?"m_5__to__m_6/MatMul/ReadVariableOp?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?#m_6__to__m_7/BiasAdd/ReadVariableOp?"m_6__to__m_7/MatMul/ReadVariableOp?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?#m_7__to__m_8/BiasAdd/ReadVariableOp?"m_7__to__m_8/MatMul/ReadVariableOp?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?#m_8__to__m_9/BiasAdd/ReadVariableOp?"m_8__to__m_9/MatMul/ReadVariableOp?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?$m_9__to__m_10/BiasAdd/ReadVariableOp?#m_9__to__m_10/MatMul/ReadVariableOp?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
"m_0__to__m_1/MatMul/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:1*
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

:1*
dtype02&
$m_14__to__m_15/MatMul/ReadVariableOp?
m_14__to__m_15/MatMulMatMul!m_13__to__m_14/Relu:activations:0,m_14__to__m_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/MatMul?
%m_14__to__m_15/BiasAdd/ReadVariableOpReadVariableOp.m_14__to__m_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%m_14__to__m_15/BiasAdd/ReadVariableOp?
m_14__to__m_15/BiasAddAddm_14__to__m_15/MatMul:product:0-m_14__to__m_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/BiasAdd?
m_14__to__m_15/SoftmaxSoftmaxm_14__to__m_15/BiasAdd:z:0*
T0*'
_output_shapes
:?????????2
m_14__to__m_15/Softmax?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp+m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2J
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
:?????????
 
_user_specified_nameinputs
?
?
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_14385593

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
?
?
__inference_loss_fn_6_14385871M
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
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_14385625

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
1__inference_m_14__to__m_15_layer_call_fn_14385794

inputs
unknown:1
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_143836112
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

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
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14384623
m_0__to__m_1_input'
m_0__to__m_1_14384457:1#
m_0__to__m_1_14384459:1'
m_1__to__m_2_14384462:11#
m_1__to__m_2_14384464:1'
m_2__to__m_3_14384467:11#
m_2__to__m_3_14384469:1'
m_3__to__m_4_14384472:11#
m_3__to__m_4_14384474:1'
m_4__to__m_5_14384477:11#
m_4__to__m_5_14384479:1'
m_5__to__m_6_14384482:11#
m_5__to__m_6_14384484:1'
m_6__to__m_7_14384487:11#
m_6__to__m_7_14384489:1'
m_7__to__m_8_14384492:11#
m_7__to__m_8_14384494:1'
m_8__to__m_9_14384497:11#
m_8__to__m_9_14384499:1(
m_9__to__m_10_14384502:11$
m_9__to__m_10_14384504:1)
m_10__to__m_11_14384507:11%
m_10__to__m_11_14384509:1)
m_11__to__m_12_14384512:11%
m_11__to__m_12_14384514:1)
m_12__to__m_13_14384517:11%
m_12__to__m_13_14384519:1)
m_13__to__m_14_14384522:11%
m_13__to__m_14_14384524:1)
m_14__to__m_15_14384527:1%
m_14__to__m_15_14384529:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallm_0__to__m_1_inputm_0__to__m_1_14384457m_0__to__m_1_14384459*
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
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_143832892&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_14384462m_1__to__m_2_14384464*
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
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_143833122&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_14384467m_2__to__m_3_14384469*
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
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_143833352&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_14384472m_3__to__m_4_14384474*
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
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_143833582&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_14384477m_4__to__m_5_14384479*
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
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_143833812&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_14384482m_5__to__m_6_14384484*
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
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_143834042&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_14384487m_6__to__m_7_14384489*
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
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_143834272&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_14384492m_7__to__m_8_14384494*
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
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_143834502&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_14384497m_8__to__m_9_14384499*
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
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_143834732&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_14384502m_9__to__m_10_14384504*
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
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_143834962'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_14384507m_10__to__m_11_14384509*
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
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_143835192(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_14384512m_11__to__m_12_14384514*
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
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_143835422(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_14384517m_12__to__m_13_14384519*
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
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_143835652(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_14384522m_13__to__m_14_14384524*
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
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_143835882(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_14384527m_14__to__m_15_14384529*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_143836112(
&m_14__to__m_15/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_14384457*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_14384462*
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
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_14384467*
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
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_14384472*
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
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_14384477*
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
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_14384482*
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
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_14384487*
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
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_14384492*
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
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_14384497*
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
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_14384502*
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
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_14384507*
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
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_14384512*
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
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_14384517*
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
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_14384522*
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_14384527*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_14385497

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
?
?
/__inference_m_7__to__m_8_layer_call_fn_14385570

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
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_143834502
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
-__inference_L-14-m_0-2_layer_call_fn_14384285
m_0__to__m_1_input
unknown:1
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

unknown_27:1

unknown_28:
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
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_143841572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
&__inference_signature_wrapper_14384786
m_0__to__m_1_input
unknown:1
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

unknown_27:1

unknown_28:
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
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference__wrapped_model_143832652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_14385529

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
??
?*
!__inference__traced_save_14386273
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
?: :1:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:1:: : : : : : : :1:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:1::1:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:11:1:1:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:1: 
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

:1: 

_output_shapes
::
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

:1: '
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

:1: C

_output_shapes
::$D 

_output_shapes

:1: E
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

:1: a

_output_shapes
::b

_output_shapes
: 
?
?
-__inference_L-14-m_0-2_layer_call_fn_14383771
m_0__to__m_1_input
unknown:1
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

unknown_27:1

unknown_28:
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
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_143837082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
0__inference_m_9__to__m_10_layer_call_fn_14385634

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
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_143834962
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
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_14383312

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
?
?
-__inference_L-14-m_0-2_layer_call_fn_14385314

inputs
unknown:1
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

unknown_27:1

unknown_28:
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
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_143841572
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
#__inference__wrapped_model_14383265
m_0__to__m_1_inputH
6l_14_m_0_2_m_0__to__m_1_matmul_readvariableop_resource:1E
7l_14_m_0_2_m_0__to__m_1_biasadd_readvariableop_resource:1H
6l_14_m_0_2_m_1__to__m_2_matmul_readvariableop_resource:11E
7l_14_m_0_2_m_1__to__m_2_biasadd_readvariableop_resource:1H
6l_14_m_0_2_m_2__to__m_3_matmul_readvariableop_resource:11E
7l_14_m_0_2_m_2__to__m_3_biasadd_readvariableop_resource:1H
6l_14_m_0_2_m_3__to__m_4_matmul_readvariableop_resource:11E
7l_14_m_0_2_m_3__to__m_4_biasadd_readvariableop_resource:1H
6l_14_m_0_2_m_4__to__m_5_matmul_readvariableop_resource:11E
7l_14_m_0_2_m_4__to__m_5_biasadd_readvariableop_resource:1H
6l_14_m_0_2_m_5__to__m_6_matmul_readvariableop_resource:11E
7l_14_m_0_2_m_5__to__m_6_biasadd_readvariableop_resource:1H
6l_14_m_0_2_m_6__to__m_7_matmul_readvariableop_resource:11E
7l_14_m_0_2_m_6__to__m_7_biasadd_readvariableop_resource:1H
6l_14_m_0_2_m_7__to__m_8_matmul_readvariableop_resource:11E
7l_14_m_0_2_m_7__to__m_8_biasadd_readvariableop_resource:1H
6l_14_m_0_2_m_8__to__m_9_matmul_readvariableop_resource:11E
7l_14_m_0_2_m_8__to__m_9_biasadd_readvariableop_resource:1I
7l_14_m_0_2_m_9__to__m_10_matmul_readvariableop_resource:11F
8l_14_m_0_2_m_9__to__m_10_biasadd_readvariableop_resource:1J
8l_14_m_0_2_m_10__to__m_11_matmul_readvariableop_resource:11G
9l_14_m_0_2_m_10__to__m_11_biasadd_readvariableop_resource:1J
8l_14_m_0_2_m_11__to__m_12_matmul_readvariableop_resource:11G
9l_14_m_0_2_m_11__to__m_12_biasadd_readvariableop_resource:1J
8l_14_m_0_2_m_12__to__m_13_matmul_readvariableop_resource:11G
9l_14_m_0_2_m_12__to__m_13_biasadd_readvariableop_resource:1J
8l_14_m_0_2_m_13__to__m_14_matmul_readvariableop_resource:11G
9l_14_m_0_2_m_13__to__m_14_biasadd_readvariableop_resource:1J
8l_14_m_0_2_m_14__to__m_15_matmul_readvariableop_resource:1G
9l_14_m_0_2_m_14__to__m_15_biasadd_readvariableop_resource:
identity??.L-14-m_0-2/m_0__to__m_1/BiasAdd/ReadVariableOp?-L-14-m_0-2/m_0__to__m_1/MatMul/ReadVariableOp?0L-14-m_0-2/m_10__to__m_11/BiasAdd/ReadVariableOp?/L-14-m_0-2/m_10__to__m_11/MatMul/ReadVariableOp?0L-14-m_0-2/m_11__to__m_12/BiasAdd/ReadVariableOp?/L-14-m_0-2/m_11__to__m_12/MatMul/ReadVariableOp?0L-14-m_0-2/m_12__to__m_13/BiasAdd/ReadVariableOp?/L-14-m_0-2/m_12__to__m_13/MatMul/ReadVariableOp?0L-14-m_0-2/m_13__to__m_14/BiasAdd/ReadVariableOp?/L-14-m_0-2/m_13__to__m_14/MatMul/ReadVariableOp?0L-14-m_0-2/m_14__to__m_15/BiasAdd/ReadVariableOp?/L-14-m_0-2/m_14__to__m_15/MatMul/ReadVariableOp?.L-14-m_0-2/m_1__to__m_2/BiasAdd/ReadVariableOp?-L-14-m_0-2/m_1__to__m_2/MatMul/ReadVariableOp?.L-14-m_0-2/m_2__to__m_3/BiasAdd/ReadVariableOp?-L-14-m_0-2/m_2__to__m_3/MatMul/ReadVariableOp?.L-14-m_0-2/m_3__to__m_4/BiasAdd/ReadVariableOp?-L-14-m_0-2/m_3__to__m_4/MatMul/ReadVariableOp?.L-14-m_0-2/m_4__to__m_5/BiasAdd/ReadVariableOp?-L-14-m_0-2/m_4__to__m_5/MatMul/ReadVariableOp?.L-14-m_0-2/m_5__to__m_6/BiasAdd/ReadVariableOp?-L-14-m_0-2/m_5__to__m_6/MatMul/ReadVariableOp?.L-14-m_0-2/m_6__to__m_7/BiasAdd/ReadVariableOp?-L-14-m_0-2/m_6__to__m_7/MatMul/ReadVariableOp?.L-14-m_0-2/m_7__to__m_8/BiasAdd/ReadVariableOp?-L-14-m_0-2/m_7__to__m_8/MatMul/ReadVariableOp?.L-14-m_0-2/m_8__to__m_9/BiasAdd/ReadVariableOp?-L-14-m_0-2/m_8__to__m_9/MatMul/ReadVariableOp?/L-14-m_0-2/m_9__to__m_10/BiasAdd/ReadVariableOp?.L-14-m_0-2/m_9__to__m_10/MatMul/ReadVariableOp?
-L-14-m_0-2/m_0__to__m_1/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_2_m_0__to__m_1_matmul_readvariableop_resource*
_output_shapes

:1*
dtype02/
-L-14-m_0-2/m_0__to__m_1/MatMul/ReadVariableOp?
L-14-m_0-2/m_0__to__m_1/MatMulMatMulm_0__to__m_1_input5L-14-m_0-2/m_0__to__m_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_0__to__m_1/MatMul?
.L-14-m_0-2/m_0__to__m_1/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_2_m_0__to__m_1_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-2/m_0__to__m_1/BiasAdd/ReadVariableOp?
L-14-m_0-2/m_0__to__m_1/BiasAddAdd(L-14-m_0-2/m_0__to__m_1/MatMul:product:06L-14-m_0-2/m_0__to__m_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-2/m_0__to__m_1/BiasAdd?
L-14-m_0-2/m_0__to__m_1/ReluRelu#L-14-m_0-2/m_0__to__m_1/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-2/m_0__to__m_1/Relu?
-L-14-m_0-2/m_1__to__m_2/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_2_m_1__to__m_2_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-2/m_1__to__m_2/MatMul/ReadVariableOp?
L-14-m_0-2/m_1__to__m_2/MatMulMatMul*L-14-m_0-2/m_0__to__m_1/Relu:activations:05L-14-m_0-2/m_1__to__m_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_1__to__m_2/MatMul?
.L-14-m_0-2/m_1__to__m_2/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_2_m_1__to__m_2_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-2/m_1__to__m_2/BiasAdd/ReadVariableOp?
L-14-m_0-2/m_1__to__m_2/BiasAddAdd(L-14-m_0-2/m_1__to__m_2/MatMul:product:06L-14-m_0-2/m_1__to__m_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-2/m_1__to__m_2/BiasAdd?
L-14-m_0-2/m_1__to__m_2/ReluRelu#L-14-m_0-2/m_1__to__m_2/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-2/m_1__to__m_2/Relu?
-L-14-m_0-2/m_2__to__m_3/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_2_m_2__to__m_3_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-2/m_2__to__m_3/MatMul/ReadVariableOp?
L-14-m_0-2/m_2__to__m_3/MatMulMatMul*L-14-m_0-2/m_1__to__m_2/Relu:activations:05L-14-m_0-2/m_2__to__m_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_2__to__m_3/MatMul?
.L-14-m_0-2/m_2__to__m_3/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_2_m_2__to__m_3_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-2/m_2__to__m_3/BiasAdd/ReadVariableOp?
L-14-m_0-2/m_2__to__m_3/BiasAddAdd(L-14-m_0-2/m_2__to__m_3/MatMul:product:06L-14-m_0-2/m_2__to__m_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-2/m_2__to__m_3/BiasAdd?
L-14-m_0-2/m_2__to__m_3/ReluRelu#L-14-m_0-2/m_2__to__m_3/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-2/m_2__to__m_3/Relu?
-L-14-m_0-2/m_3__to__m_4/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_2_m_3__to__m_4_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-2/m_3__to__m_4/MatMul/ReadVariableOp?
L-14-m_0-2/m_3__to__m_4/MatMulMatMul*L-14-m_0-2/m_2__to__m_3/Relu:activations:05L-14-m_0-2/m_3__to__m_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_3__to__m_4/MatMul?
.L-14-m_0-2/m_3__to__m_4/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_2_m_3__to__m_4_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-2/m_3__to__m_4/BiasAdd/ReadVariableOp?
L-14-m_0-2/m_3__to__m_4/BiasAddAdd(L-14-m_0-2/m_3__to__m_4/MatMul:product:06L-14-m_0-2/m_3__to__m_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-2/m_3__to__m_4/BiasAdd?
L-14-m_0-2/m_3__to__m_4/ReluRelu#L-14-m_0-2/m_3__to__m_4/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-2/m_3__to__m_4/Relu?
-L-14-m_0-2/m_4__to__m_5/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_2_m_4__to__m_5_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-2/m_4__to__m_5/MatMul/ReadVariableOp?
L-14-m_0-2/m_4__to__m_5/MatMulMatMul*L-14-m_0-2/m_3__to__m_4/Relu:activations:05L-14-m_0-2/m_4__to__m_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_4__to__m_5/MatMul?
.L-14-m_0-2/m_4__to__m_5/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_2_m_4__to__m_5_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-2/m_4__to__m_5/BiasAdd/ReadVariableOp?
L-14-m_0-2/m_4__to__m_5/BiasAddAdd(L-14-m_0-2/m_4__to__m_5/MatMul:product:06L-14-m_0-2/m_4__to__m_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-2/m_4__to__m_5/BiasAdd?
L-14-m_0-2/m_4__to__m_5/ReluRelu#L-14-m_0-2/m_4__to__m_5/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-2/m_4__to__m_5/Relu?
-L-14-m_0-2/m_5__to__m_6/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_2_m_5__to__m_6_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-2/m_5__to__m_6/MatMul/ReadVariableOp?
L-14-m_0-2/m_5__to__m_6/MatMulMatMul*L-14-m_0-2/m_4__to__m_5/Relu:activations:05L-14-m_0-2/m_5__to__m_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_5__to__m_6/MatMul?
.L-14-m_0-2/m_5__to__m_6/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_2_m_5__to__m_6_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-2/m_5__to__m_6/BiasAdd/ReadVariableOp?
L-14-m_0-2/m_5__to__m_6/BiasAddAdd(L-14-m_0-2/m_5__to__m_6/MatMul:product:06L-14-m_0-2/m_5__to__m_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-2/m_5__to__m_6/BiasAdd?
L-14-m_0-2/m_5__to__m_6/ReluRelu#L-14-m_0-2/m_5__to__m_6/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-2/m_5__to__m_6/Relu?
-L-14-m_0-2/m_6__to__m_7/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_2_m_6__to__m_7_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-2/m_6__to__m_7/MatMul/ReadVariableOp?
L-14-m_0-2/m_6__to__m_7/MatMulMatMul*L-14-m_0-2/m_5__to__m_6/Relu:activations:05L-14-m_0-2/m_6__to__m_7/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_6__to__m_7/MatMul?
.L-14-m_0-2/m_6__to__m_7/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_2_m_6__to__m_7_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-2/m_6__to__m_7/BiasAdd/ReadVariableOp?
L-14-m_0-2/m_6__to__m_7/BiasAddAdd(L-14-m_0-2/m_6__to__m_7/MatMul:product:06L-14-m_0-2/m_6__to__m_7/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-2/m_6__to__m_7/BiasAdd?
L-14-m_0-2/m_6__to__m_7/ReluRelu#L-14-m_0-2/m_6__to__m_7/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-2/m_6__to__m_7/Relu?
-L-14-m_0-2/m_7__to__m_8/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_2_m_7__to__m_8_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-2/m_7__to__m_8/MatMul/ReadVariableOp?
L-14-m_0-2/m_7__to__m_8/MatMulMatMul*L-14-m_0-2/m_6__to__m_7/Relu:activations:05L-14-m_0-2/m_7__to__m_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_7__to__m_8/MatMul?
.L-14-m_0-2/m_7__to__m_8/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_2_m_7__to__m_8_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-2/m_7__to__m_8/BiasAdd/ReadVariableOp?
L-14-m_0-2/m_7__to__m_8/BiasAddAdd(L-14-m_0-2/m_7__to__m_8/MatMul:product:06L-14-m_0-2/m_7__to__m_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-2/m_7__to__m_8/BiasAdd?
L-14-m_0-2/m_7__to__m_8/ReluRelu#L-14-m_0-2/m_7__to__m_8/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-2/m_7__to__m_8/Relu?
-L-14-m_0-2/m_8__to__m_9/MatMul/ReadVariableOpReadVariableOp6l_14_m_0_2_m_8__to__m_9_matmul_readvariableop_resource*
_output_shapes

:11*
dtype02/
-L-14-m_0-2/m_8__to__m_9/MatMul/ReadVariableOp?
L-14-m_0-2/m_8__to__m_9/MatMulMatMul*L-14-m_0-2/m_7__to__m_8/Relu:activations:05L-14-m_0-2/m_8__to__m_9/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_8__to__m_9/MatMul?
.L-14-m_0-2/m_8__to__m_9/BiasAdd/ReadVariableOpReadVariableOp7l_14_m_0_2_m_8__to__m_9_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype020
.L-14-m_0-2/m_8__to__m_9/BiasAdd/ReadVariableOp?
L-14-m_0-2/m_8__to__m_9/BiasAddAdd(L-14-m_0-2/m_8__to__m_9/MatMul:product:06L-14-m_0-2/m_8__to__m_9/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-2/m_8__to__m_9/BiasAdd?
L-14-m_0-2/m_8__to__m_9/ReluRelu#L-14-m_0-2/m_8__to__m_9/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-2/m_8__to__m_9/Relu?
.L-14-m_0-2/m_9__to__m_10/MatMul/ReadVariableOpReadVariableOp7l_14_m_0_2_m_9__to__m_10_matmul_readvariableop_resource*
_output_shapes

:11*
dtype020
.L-14-m_0-2/m_9__to__m_10/MatMul/ReadVariableOp?
L-14-m_0-2/m_9__to__m_10/MatMulMatMul*L-14-m_0-2/m_8__to__m_9/Relu:activations:06L-14-m_0-2/m_9__to__m_10/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12!
L-14-m_0-2/m_9__to__m_10/MatMul?
/L-14-m_0-2/m_9__to__m_10/BiasAdd/ReadVariableOpReadVariableOp8l_14_m_0_2_m_9__to__m_10_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype021
/L-14-m_0-2/m_9__to__m_10/BiasAdd/ReadVariableOp?
 L-14-m_0-2/m_9__to__m_10/BiasAddAdd)L-14-m_0-2/m_9__to__m_10/MatMul:product:07L-14-m_0-2/m_9__to__m_10/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12"
 L-14-m_0-2/m_9__to__m_10/BiasAdd?
L-14-m_0-2/m_9__to__m_10/ReluRelu$L-14-m_0-2/m_9__to__m_10/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12
L-14-m_0-2/m_9__to__m_10/Relu?
/L-14-m_0-2/m_10__to__m_11/MatMul/ReadVariableOpReadVariableOp8l_14_m_0_2_m_10__to__m_11_matmul_readvariableop_resource*
_output_shapes

:11*
dtype021
/L-14-m_0-2/m_10__to__m_11/MatMul/ReadVariableOp?
 L-14-m_0-2/m_10__to__m_11/MatMulMatMul+L-14-m_0-2/m_9__to__m_10/Relu:activations:07L-14-m_0-2/m_10__to__m_11/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12"
 L-14-m_0-2/m_10__to__m_11/MatMul?
0L-14-m_0-2/m_10__to__m_11/BiasAdd/ReadVariableOpReadVariableOp9l_14_m_0_2_m_10__to__m_11_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype022
0L-14-m_0-2/m_10__to__m_11/BiasAdd/ReadVariableOp?
!L-14-m_0-2/m_10__to__m_11/BiasAddAdd*L-14-m_0-2/m_10__to__m_11/MatMul:product:08L-14-m_0-2/m_10__to__m_11/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12#
!L-14-m_0-2/m_10__to__m_11/BiasAdd?
L-14-m_0-2/m_10__to__m_11/ReluRelu%L-14-m_0-2/m_10__to__m_11/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_10__to__m_11/Relu?
/L-14-m_0-2/m_11__to__m_12/MatMul/ReadVariableOpReadVariableOp8l_14_m_0_2_m_11__to__m_12_matmul_readvariableop_resource*
_output_shapes

:11*
dtype021
/L-14-m_0-2/m_11__to__m_12/MatMul/ReadVariableOp?
 L-14-m_0-2/m_11__to__m_12/MatMulMatMul,L-14-m_0-2/m_10__to__m_11/Relu:activations:07L-14-m_0-2/m_11__to__m_12/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12"
 L-14-m_0-2/m_11__to__m_12/MatMul?
0L-14-m_0-2/m_11__to__m_12/BiasAdd/ReadVariableOpReadVariableOp9l_14_m_0_2_m_11__to__m_12_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype022
0L-14-m_0-2/m_11__to__m_12/BiasAdd/ReadVariableOp?
!L-14-m_0-2/m_11__to__m_12/BiasAddAdd*L-14-m_0-2/m_11__to__m_12/MatMul:product:08L-14-m_0-2/m_11__to__m_12/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12#
!L-14-m_0-2/m_11__to__m_12/BiasAdd?
L-14-m_0-2/m_11__to__m_12/ReluRelu%L-14-m_0-2/m_11__to__m_12/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_11__to__m_12/Relu?
/L-14-m_0-2/m_12__to__m_13/MatMul/ReadVariableOpReadVariableOp8l_14_m_0_2_m_12__to__m_13_matmul_readvariableop_resource*
_output_shapes

:11*
dtype021
/L-14-m_0-2/m_12__to__m_13/MatMul/ReadVariableOp?
 L-14-m_0-2/m_12__to__m_13/MatMulMatMul,L-14-m_0-2/m_11__to__m_12/Relu:activations:07L-14-m_0-2/m_12__to__m_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12"
 L-14-m_0-2/m_12__to__m_13/MatMul?
0L-14-m_0-2/m_12__to__m_13/BiasAdd/ReadVariableOpReadVariableOp9l_14_m_0_2_m_12__to__m_13_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype022
0L-14-m_0-2/m_12__to__m_13/BiasAdd/ReadVariableOp?
!L-14-m_0-2/m_12__to__m_13/BiasAddAdd*L-14-m_0-2/m_12__to__m_13/MatMul:product:08L-14-m_0-2/m_12__to__m_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12#
!L-14-m_0-2/m_12__to__m_13/BiasAdd?
L-14-m_0-2/m_12__to__m_13/ReluRelu%L-14-m_0-2/m_12__to__m_13/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_12__to__m_13/Relu?
/L-14-m_0-2/m_13__to__m_14/MatMul/ReadVariableOpReadVariableOp8l_14_m_0_2_m_13__to__m_14_matmul_readvariableop_resource*
_output_shapes

:11*
dtype021
/L-14-m_0-2/m_13__to__m_14/MatMul/ReadVariableOp?
 L-14-m_0-2/m_13__to__m_14/MatMulMatMul,L-14-m_0-2/m_12__to__m_13/Relu:activations:07L-14-m_0-2/m_13__to__m_14/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12"
 L-14-m_0-2/m_13__to__m_14/MatMul?
0L-14-m_0-2/m_13__to__m_14/BiasAdd/ReadVariableOpReadVariableOp9l_14_m_0_2_m_13__to__m_14_biasadd_readvariableop_resource*
_output_shapes
:1*
dtype022
0L-14-m_0-2/m_13__to__m_14/BiasAdd/ReadVariableOp?
!L-14-m_0-2/m_13__to__m_14/BiasAddAdd*L-14-m_0-2/m_13__to__m_14/MatMul:product:08L-14-m_0-2/m_13__to__m_14/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????12#
!L-14-m_0-2/m_13__to__m_14/BiasAdd?
L-14-m_0-2/m_13__to__m_14/ReluRelu%L-14-m_0-2/m_13__to__m_14/BiasAdd:z:0*
T0*'
_output_shapes
:?????????12 
L-14-m_0-2/m_13__to__m_14/Relu?
/L-14-m_0-2/m_14__to__m_15/MatMul/ReadVariableOpReadVariableOp8l_14_m_0_2_m_14__to__m_15_matmul_readvariableop_resource*
_output_shapes

:1*
dtype021
/L-14-m_0-2/m_14__to__m_15/MatMul/ReadVariableOp?
 L-14-m_0-2/m_14__to__m_15/MatMulMatMul,L-14-m_0-2/m_13__to__m_14/Relu:activations:07L-14-m_0-2/m_14__to__m_15/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 L-14-m_0-2/m_14__to__m_15/MatMul?
0L-14-m_0-2/m_14__to__m_15/BiasAdd/ReadVariableOpReadVariableOp9l_14_m_0_2_m_14__to__m_15_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0L-14-m_0-2/m_14__to__m_15/BiasAdd/ReadVariableOp?
!L-14-m_0-2/m_14__to__m_15/BiasAddAdd*L-14-m_0-2/m_14__to__m_15/MatMul:product:08L-14-m_0-2/m_14__to__m_15/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!L-14-m_0-2/m_14__to__m_15/BiasAdd?
!L-14-m_0-2/m_14__to__m_15/SoftmaxSoftmax%L-14-m_0-2/m_14__to__m_15/BiasAdd:z:0*
T0*'
_output_shapes
:?????????2#
!L-14-m_0-2/m_14__to__m_15/Softmax?
IdentityIdentity+L-14-m_0-2/m_14__to__m_15/Softmax:softmax:0/^L-14-m_0-2/m_0__to__m_1/BiasAdd/ReadVariableOp.^L-14-m_0-2/m_0__to__m_1/MatMul/ReadVariableOp1^L-14-m_0-2/m_10__to__m_11/BiasAdd/ReadVariableOp0^L-14-m_0-2/m_10__to__m_11/MatMul/ReadVariableOp1^L-14-m_0-2/m_11__to__m_12/BiasAdd/ReadVariableOp0^L-14-m_0-2/m_11__to__m_12/MatMul/ReadVariableOp1^L-14-m_0-2/m_12__to__m_13/BiasAdd/ReadVariableOp0^L-14-m_0-2/m_12__to__m_13/MatMul/ReadVariableOp1^L-14-m_0-2/m_13__to__m_14/BiasAdd/ReadVariableOp0^L-14-m_0-2/m_13__to__m_14/MatMul/ReadVariableOp1^L-14-m_0-2/m_14__to__m_15/BiasAdd/ReadVariableOp0^L-14-m_0-2/m_14__to__m_15/MatMul/ReadVariableOp/^L-14-m_0-2/m_1__to__m_2/BiasAdd/ReadVariableOp.^L-14-m_0-2/m_1__to__m_2/MatMul/ReadVariableOp/^L-14-m_0-2/m_2__to__m_3/BiasAdd/ReadVariableOp.^L-14-m_0-2/m_2__to__m_3/MatMul/ReadVariableOp/^L-14-m_0-2/m_3__to__m_4/BiasAdd/ReadVariableOp.^L-14-m_0-2/m_3__to__m_4/MatMul/ReadVariableOp/^L-14-m_0-2/m_4__to__m_5/BiasAdd/ReadVariableOp.^L-14-m_0-2/m_4__to__m_5/MatMul/ReadVariableOp/^L-14-m_0-2/m_5__to__m_6/BiasAdd/ReadVariableOp.^L-14-m_0-2/m_5__to__m_6/MatMul/ReadVariableOp/^L-14-m_0-2/m_6__to__m_7/BiasAdd/ReadVariableOp.^L-14-m_0-2/m_6__to__m_7/MatMul/ReadVariableOp/^L-14-m_0-2/m_7__to__m_8/BiasAdd/ReadVariableOp.^L-14-m_0-2/m_7__to__m_8/MatMul/ReadVariableOp/^L-14-m_0-2/m_8__to__m_9/BiasAdd/ReadVariableOp.^L-14-m_0-2/m_8__to__m_9/MatMul/ReadVariableOp0^L-14-m_0-2/m_9__to__m_10/BiasAdd/ReadVariableOp/^L-14-m_0-2/m_9__to__m_10/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.L-14-m_0-2/m_0__to__m_1/BiasAdd/ReadVariableOp.L-14-m_0-2/m_0__to__m_1/BiasAdd/ReadVariableOp2^
-L-14-m_0-2/m_0__to__m_1/MatMul/ReadVariableOp-L-14-m_0-2/m_0__to__m_1/MatMul/ReadVariableOp2d
0L-14-m_0-2/m_10__to__m_11/BiasAdd/ReadVariableOp0L-14-m_0-2/m_10__to__m_11/BiasAdd/ReadVariableOp2b
/L-14-m_0-2/m_10__to__m_11/MatMul/ReadVariableOp/L-14-m_0-2/m_10__to__m_11/MatMul/ReadVariableOp2d
0L-14-m_0-2/m_11__to__m_12/BiasAdd/ReadVariableOp0L-14-m_0-2/m_11__to__m_12/BiasAdd/ReadVariableOp2b
/L-14-m_0-2/m_11__to__m_12/MatMul/ReadVariableOp/L-14-m_0-2/m_11__to__m_12/MatMul/ReadVariableOp2d
0L-14-m_0-2/m_12__to__m_13/BiasAdd/ReadVariableOp0L-14-m_0-2/m_12__to__m_13/BiasAdd/ReadVariableOp2b
/L-14-m_0-2/m_12__to__m_13/MatMul/ReadVariableOp/L-14-m_0-2/m_12__to__m_13/MatMul/ReadVariableOp2d
0L-14-m_0-2/m_13__to__m_14/BiasAdd/ReadVariableOp0L-14-m_0-2/m_13__to__m_14/BiasAdd/ReadVariableOp2b
/L-14-m_0-2/m_13__to__m_14/MatMul/ReadVariableOp/L-14-m_0-2/m_13__to__m_14/MatMul/ReadVariableOp2d
0L-14-m_0-2/m_14__to__m_15/BiasAdd/ReadVariableOp0L-14-m_0-2/m_14__to__m_15/BiasAdd/ReadVariableOp2b
/L-14-m_0-2/m_14__to__m_15/MatMul/ReadVariableOp/L-14-m_0-2/m_14__to__m_15/MatMul/ReadVariableOp2`
.L-14-m_0-2/m_1__to__m_2/BiasAdd/ReadVariableOp.L-14-m_0-2/m_1__to__m_2/BiasAdd/ReadVariableOp2^
-L-14-m_0-2/m_1__to__m_2/MatMul/ReadVariableOp-L-14-m_0-2/m_1__to__m_2/MatMul/ReadVariableOp2`
.L-14-m_0-2/m_2__to__m_3/BiasAdd/ReadVariableOp.L-14-m_0-2/m_2__to__m_3/BiasAdd/ReadVariableOp2^
-L-14-m_0-2/m_2__to__m_3/MatMul/ReadVariableOp-L-14-m_0-2/m_2__to__m_3/MatMul/ReadVariableOp2`
.L-14-m_0-2/m_3__to__m_4/BiasAdd/ReadVariableOp.L-14-m_0-2/m_3__to__m_4/BiasAdd/ReadVariableOp2^
-L-14-m_0-2/m_3__to__m_4/MatMul/ReadVariableOp-L-14-m_0-2/m_3__to__m_4/MatMul/ReadVariableOp2`
.L-14-m_0-2/m_4__to__m_5/BiasAdd/ReadVariableOp.L-14-m_0-2/m_4__to__m_5/BiasAdd/ReadVariableOp2^
-L-14-m_0-2/m_4__to__m_5/MatMul/ReadVariableOp-L-14-m_0-2/m_4__to__m_5/MatMul/ReadVariableOp2`
.L-14-m_0-2/m_5__to__m_6/BiasAdd/ReadVariableOp.L-14-m_0-2/m_5__to__m_6/BiasAdd/ReadVariableOp2^
-L-14-m_0-2/m_5__to__m_6/MatMul/ReadVariableOp-L-14-m_0-2/m_5__to__m_6/MatMul/ReadVariableOp2`
.L-14-m_0-2/m_6__to__m_7/BiasAdd/ReadVariableOp.L-14-m_0-2/m_6__to__m_7/BiasAdd/ReadVariableOp2^
-L-14-m_0-2/m_6__to__m_7/MatMul/ReadVariableOp-L-14-m_0-2/m_6__to__m_7/MatMul/ReadVariableOp2`
.L-14-m_0-2/m_7__to__m_8/BiasAdd/ReadVariableOp.L-14-m_0-2/m_7__to__m_8/BiasAdd/ReadVariableOp2^
-L-14-m_0-2/m_7__to__m_8/MatMul/ReadVariableOp-L-14-m_0-2/m_7__to__m_8/MatMul/ReadVariableOp2`
.L-14-m_0-2/m_8__to__m_9/BiasAdd/ReadVariableOp.L-14-m_0-2/m_8__to__m_9/BiasAdd/ReadVariableOp2^
-L-14-m_0-2/m_8__to__m_9/MatMul/ReadVariableOp-L-14-m_0-2/m_8__to__m_9/MatMul/ReadVariableOp2b
/L-14-m_0-2/m_9__to__m_10/BiasAdd/ReadVariableOp/L-14-m_0-2/m_9__to__m_10/BiasAdd/ReadVariableOp2`
.L-14-m_0-2/m_9__to__m_10/MatMul/ReadVariableOp.L-14-m_0-2/m_9__to__m_10/MatMul/ReadVariableOp:[ W
'
_output_shapes
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
-__inference_L-14-m_0-2_layer_call_fn_14385249

inputs
unknown:1
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

unknown_27:1

unknown_28:
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
:?????????*@
_read_only_resource_inputs"
 	
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_143837082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_14383404

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
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_14383427

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
?
?
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_14385561

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
?
?
__inference_loss_fn_4_14385849M
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
?
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14384454
m_0__to__m_1_input'
m_0__to__m_1_14384288:1#
m_0__to__m_1_14384290:1'
m_1__to__m_2_14384293:11#
m_1__to__m_2_14384295:1'
m_2__to__m_3_14384298:11#
m_2__to__m_3_14384300:1'
m_3__to__m_4_14384303:11#
m_3__to__m_4_14384305:1'
m_4__to__m_5_14384308:11#
m_4__to__m_5_14384310:1'
m_5__to__m_6_14384313:11#
m_5__to__m_6_14384315:1'
m_6__to__m_7_14384318:11#
m_6__to__m_7_14384320:1'
m_7__to__m_8_14384323:11#
m_7__to__m_8_14384325:1'
m_8__to__m_9_14384328:11#
m_8__to__m_9_14384330:1(
m_9__to__m_10_14384333:11$
m_9__to__m_10_14384335:1)
m_10__to__m_11_14384338:11%
m_10__to__m_11_14384340:1)
m_11__to__m_12_14384343:11%
m_11__to__m_12_14384345:1)
m_12__to__m_13_14384348:11%
m_12__to__m_13_14384350:1)
m_13__to__m_14_14384353:11%
m_13__to__m_14_14384355:1)
m_14__to__m_15_14384358:1%
m_14__to__m_15_14384360:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallm_0__to__m_1_inputm_0__to__m_1_14384288m_0__to__m_1_14384290*
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
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_143832892&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_14384293m_1__to__m_2_14384295*
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
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_143833122&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_14384298m_2__to__m_3_14384300*
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
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_143833352&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_14384303m_3__to__m_4_14384305*
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
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_143833582&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_14384308m_4__to__m_5_14384310*
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
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_143833812&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_14384313m_5__to__m_6_14384315*
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
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_143834042&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_14384318m_6__to__m_7_14384320*
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
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_143834272&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_14384323m_7__to__m_8_14384325*
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
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_143834502&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_14384328m_8__to__m_9_14384330*
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
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_143834732&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_14384333m_9__to__m_10_14384335*
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
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_143834962'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_14384338m_10__to__m_11_14384340*
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
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_143835192(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_14384343m_11__to__m_12_14384345*
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
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_143835422(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_14384348m_12__to__m_13_14384350*
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
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_143835652(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_14384353m_13__to__m_14_14384355*
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
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_143835882(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_14384358m_14__to__m_15_14384360*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_143836112(
&m_14__to__m_15/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_14384288*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_14384293*
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
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_14384298*
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
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_14384303*
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
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_14384308*
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
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_14384313*
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
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_14384318*
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
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_14384323*
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
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_14384328*
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
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_14384333*
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
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_14384338*
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
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_14384343*
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
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_14384348*
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
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_14384353*
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_14384358*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
:?????????
,
_user_specified_namem_0__to__m_1_input
?
?
__inference_loss_fn_8_14385893M
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
??
?
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14384157

inputs'
m_0__to__m_1_14383991:1#
m_0__to__m_1_14383993:1'
m_1__to__m_2_14383996:11#
m_1__to__m_2_14383998:1'
m_2__to__m_3_14384001:11#
m_2__to__m_3_14384003:1'
m_3__to__m_4_14384006:11#
m_3__to__m_4_14384008:1'
m_4__to__m_5_14384011:11#
m_4__to__m_5_14384013:1'
m_5__to__m_6_14384016:11#
m_5__to__m_6_14384018:1'
m_6__to__m_7_14384021:11#
m_6__to__m_7_14384023:1'
m_7__to__m_8_14384026:11#
m_7__to__m_8_14384028:1'
m_8__to__m_9_14384031:11#
m_8__to__m_9_14384033:1(
m_9__to__m_10_14384036:11$
m_9__to__m_10_14384038:1)
m_10__to__m_11_14384041:11%
m_10__to__m_11_14384043:1)
m_11__to__m_12_14384046:11%
m_11__to__m_12_14384048:1)
m_12__to__m_13_14384051:11%
m_12__to__m_13_14384053:1)
m_13__to__m_14_14384056:11%
m_13__to__m_14_14384058:1)
m_14__to__m_15_14384061:1%
m_14__to__m_15_14384063:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallinputsm_0__to__m_1_14383991m_0__to__m_1_14383993*
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
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_143832892&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_14383996m_1__to__m_2_14383998*
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
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_143833122&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_14384001m_2__to__m_3_14384003*
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
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_143833352&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_14384006m_3__to__m_4_14384008*
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
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_143833582&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_14384011m_4__to__m_5_14384013*
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
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_143833812&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_14384016m_5__to__m_6_14384018*
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
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_143834042&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_14384021m_6__to__m_7_14384023*
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
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_143834272&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_14384026m_7__to__m_8_14384028*
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
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_143834502&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_14384031m_8__to__m_9_14384033*
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
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_143834732&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_14384036m_9__to__m_10_14384038*
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
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_143834962'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_14384041m_10__to__m_11_14384043*
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
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_143835192(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_14384046m_11__to__m_12_14384048*
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
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_143835422(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_14384051m_12__to__m_13_14384053*
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
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_143835652(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_14384056m_13__to__m_14_14384058*
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
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_143835882(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_14384061m_14__to__m_15_14384063*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_143836112(
&m_14__to__m_15/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_14383991*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_14383996*
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
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_14384001*
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
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_14384006*
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
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_14384011*
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
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_14384016*
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
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_14384021*
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
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_14384026*
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
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_14384031*
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
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_14384036*
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
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_14384041*
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
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_14384046*
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
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_14384051*
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
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_14384056*
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_14384061*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
:?????????
 
_user_specified_nameinputs
?
?
1__inference_m_11__to__m_12_layer_call_fn_14385698

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
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_143835422
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
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_14385465

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
?
?
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_14385785

inputs0
matmul_readvariableop_resource:1-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd\
SoftmaxSoftmaxBiasAdd:z:0*
T0*'
_output_shapes
:?????????2	
Softmax?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
:?????????2

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
__inference_loss_fn_12_14385937O
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
?
?
__inference_loss_fn_7_14385882M
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
?
?
__inference_loss_fn_10_14385915O
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
?
?
/__inference_m_1__to__m_2_layer_call_fn_14385378

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
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_143833122
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
/__inference_m_4__to__m_5_layer_call_fn_14385474

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
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_143833812
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
/__inference_m_0__to__m_1_layer_call_fn_14385346

inputs
unknown:1
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
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_143832892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_14385433

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
?
?
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_14385721

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
?
?
__inference_loss_fn_14_14385959O
=m_14__to__m_15_kernel_regularizer_abs_readvariableop_resource:1
identity??4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp=m_14__to__m_15_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
?
?
__inference_loss_fn_9_14385904N
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
?
?
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_14383289

inputs0
matmul_readvariableop_resource:1-
biasadd_readvariableop_resource:1
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
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

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2h
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_14383473

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
??
?>
$__inference__traced_restore_14386574
file_prefix6
$assignvariableop_m_0__to__m_1_kernel:12
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
)assignvariableop_28_m_14__to__m_15_kernel:15
'assignvariableop_29_m_14__to__m_15_bias:'
assignvariableop_30_adam_iter:	 )
assignvariableop_31_adam_beta_1: )
assignvariableop_32_adam_beta_2: (
assignvariableop_33_adam_decay: 0
&assignvariableop_34_adam_learning_rate: #
assignvariableop_35_total: #
assignvariableop_36_count: @
.assignvariableop_37_adam_m_0__to__m_1_kernel_m:1:
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
0assignvariableop_65_adam_m_14__to__m_15_kernel_m:1<
.assignvariableop_66_adam_m_14__to__m_15_bias_m:@
.assignvariableop_67_adam_m_0__to__m_1_kernel_v:1:
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
0assignvariableop_95_adam_m_14__to__m_15_kernel_v:1<
.assignvariableop_96_adam_m_14__to__m_15_bias_v:
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
/__inference_m_6__to__m_7_layer_call_fn_14385538

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
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_143834272
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
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_14385753

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
?
?
__inference_loss_fn_13_14385948O
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
?
?
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_14383565

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
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_14383611

inputs0
matmul_readvariableop_resource:1-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp}
BiasAddAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd\
SoftmaxSoftmaxBiasAdd:z:0*
T0*'
_output_shapes
:?????????2	
Softmax?
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
:?????????2

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
??
?
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14383708

inputs'
m_0__to__m_1_14383290:1#
m_0__to__m_1_14383292:1'
m_1__to__m_2_14383313:11#
m_1__to__m_2_14383315:1'
m_2__to__m_3_14383336:11#
m_2__to__m_3_14383338:1'
m_3__to__m_4_14383359:11#
m_3__to__m_4_14383361:1'
m_4__to__m_5_14383382:11#
m_4__to__m_5_14383384:1'
m_5__to__m_6_14383405:11#
m_5__to__m_6_14383407:1'
m_6__to__m_7_14383428:11#
m_6__to__m_7_14383430:1'
m_7__to__m_8_14383451:11#
m_7__to__m_8_14383453:1'
m_8__to__m_9_14383474:11#
m_8__to__m_9_14383476:1(
m_9__to__m_10_14383497:11$
m_9__to__m_10_14383499:1)
m_10__to__m_11_14383520:11%
m_10__to__m_11_14383522:1)
m_11__to__m_12_14383543:11%
m_11__to__m_12_14383545:1)
m_12__to__m_13_14383566:11%
m_12__to__m_13_14383568:1)
m_13__to__m_14_14383589:11%
m_13__to__m_14_14383591:1)
m_14__to__m_15_14383612:1%
m_14__to__m_15_14383614:
identity??$m_0__to__m_1/StatefulPartitionedCall?2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?&m_10__to__m_11/StatefulPartitionedCall?4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOp?&m_11__to__m_12/StatefulPartitionedCall?4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOp?&m_12__to__m_13/StatefulPartitionedCall?4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOp?&m_13__to__m_14/StatefulPartitionedCall?4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOp?&m_14__to__m_15/StatefulPartitionedCall?4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?$m_1__to__m_2/StatefulPartitionedCall?2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOp?$m_2__to__m_3/StatefulPartitionedCall?2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOp?$m_3__to__m_4/StatefulPartitionedCall?2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOp?$m_4__to__m_5/StatefulPartitionedCall?2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOp?$m_5__to__m_6/StatefulPartitionedCall?2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOp?$m_6__to__m_7/StatefulPartitionedCall?2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOp?$m_7__to__m_8/StatefulPartitionedCall?2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOp?$m_8__to__m_9/StatefulPartitionedCall?2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOp?%m_9__to__m_10/StatefulPartitionedCall?3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOp?
$m_0__to__m_1/StatefulPartitionedCallStatefulPartitionedCallinputsm_0__to__m_1_14383290m_0__to__m_1_14383292*
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
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_143832892&
$m_0__to__m_1/StatefulPartitionedCall?
$m_1__to__m_2/StatefulPartitionedCallStatefulPartitionedCall-m_0__to__m_1/StatefulPartitionedCall:output:0m_1__to__m_2_14383313m_1__to__m_2_14383315*
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
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_143833122&
$m_1__to__m_2/StatefulPartitionedCall?
$m_2__to__m_3/StatefulPartitionedCallStatefulPartitionedCall-m_1__to__m_2/StatefulPartitionedCall:output:0m_2__to__m_3_14383336m_2__to__m_3_14383338*
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
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_143833352&
$m_2__to__m_3/StatefulPartitionedCall?
$m_3__to__m_4/StatefulPartitionedCallStatefulPartitionedCall-m_2__to__m_3/StatefulPartitionedCall:output:0m_3__to__m_4_14383359m_3__to__m_4_14383361*
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
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_143833582&
$m_3__to__m_4/StatefulPartitionedCall?
$m_4__to__m_5/StatefulPartitionedCallStatefulPartitionedCall-m_3__to__m_4/StatefulPartitionedCall:output:0m_4__to__m_5_14383382m_4__to__m_5_14383384*
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
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_143833812&
$m_4__to__m_5/StatefulPartitionedCall?
$m_5__to__m_6/StatefulPartitionedCallStatefulPartitionedCall-m_4__to__m_5/StatefulPartitionedCall:output:0m_5__to__m_6_14383405m_5__to__m_6_14383407*
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
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_143834042&
$m_5__to__m_6/StatefulPartitionedCall?
$m_6__to__m_7/StatefulPartitionedCallStatefulPartitionedCall-m_5__to__m_6/StatefulPartitionedCall:output:0m_6__to__m_7_14383428m_6__to__m_7_14383430*
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
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_143834272&
$m_6__to__m_7/StatefulPartitionedCall?
$m_7__to__m_8/StatefulPartitionedCallStatefulPartitionedCall-m_6__to__m_7/StatefulPartitionedCall:output:0m_7__to__m_8_14383451m_7__to__m_8_14383453*
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
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_143834502&
$m_7__to__m_8/StatefulPartitionedCall?
$m_8__to__m_9/StatefulPartitionedCallStatefulPartitionedCall-m_7__to__m_8/StatefulPartitionedCall:output:0m_8__to__m_9_14383474m_8__to__m_9_14383476*
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
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_143834732&
$m_8__to__m_9/StatefulPartitionedCall?
%m_9__to__m_10/StatefulPartitionedCallStatefulPartitionedCall-m_8__to__m_9/StatefulPartitionedCall:output:0m_9__to__m_10_14383497m_9__to__m_10_14383499*
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
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_143834962'
%m_9__to__m_10/StatefulPartitionedCall?
&m_10__to__m_11/StatefulPartitionedCallStatefulPartitionedCall.m_9__to__m_10/StatefulPartitionedCall:output:0m_10__to__m_11_14383520m_10__to__m_11_14383522*
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
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_143835192(
&m_10__to__m_11/StatefulPartitionedCall?
&m_11__to__m_12/StatefulPartitionedCallStatefulPartitionedCall/m_10__to__m_11/StatefulPartitionedCall:output:0m_11__to__m_12_14383543m_11__to__m_12_14383545*
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
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_143835422(
&m_11__to__m_12/StatefulPartitionedCall?
&m_12__to__m_13/StatefulPartitionedCallStatefulPartitionedCall/m_11__to__m_12/StatefulPartitionedCall:output:0m_12__to__m_13_14383566m_12__to__m_13_14383568*
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
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_143835652(
&m_12__to__m_13/StatefulPartitionedCall?
&m_13__to__m_14/StatefulPartitionedCallStatefulPartitionedCall/m_12__to__m_13/StatefulPartitionedCall:output:0m_13__to__m_14_14383589m_13__to__m_14_14383591*
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
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_143835882(
&m_13__to__m_14/StatefulPartitionedCall?
&m_14__to__m_15/StatefulPartitionedCallStatefulPartitionedCall/m_13__to__m_14/StatefulPartitionedCall:output:0m_14__to__m_15_14383612m_14__to__m_15_14383614*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_143836112(
&m_14__to__m_15/StatefulPartitionedCall?
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_0__to__m_1_14383290*
_output_shapes

:1*
dtype024
2m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp?
#m_0__to__m_1/kernel/Regularizer/AbsAbs:m_0__to__m_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12%
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
2m_1__to__m_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_1__to__m_2_14383313*
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
2m_2__to__m_3/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_2__to__m_3_14383336*
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
2m_3__to__m_4/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_3__to__m_4_14383359*
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
2m_4__to__m_5/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_4__to__m_5_14383382*
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
2m_5__to__m_6/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_5__to__m_6_14383405*
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
2m_6__to__m_7/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_6__to__m_7_14383428*
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
2m_7__to__m_8/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_7__to__m_8_14383451*
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
2m_8__to__m_9/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_8__to__m_9_14383474*
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
3m_9__to__m_10/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_9__to__m_10_14383497*
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
4m_10__to__m_11/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_10__to__m_11_14383520*
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
4m_11__to__m_12/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_11__to__m_12_14383543*
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
4m_12__to__m_13/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_12__to__m_13_14383566*
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
4m_13__to__m_14/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_13__to__m_14_14383589*
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
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpm_14__to__m_15_14383612*
_output_shapes

:1*
dtype026
4m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp?
%m_14__to__m_15/kernel/Regularizer/AbsAbs<m_14__to__m_15/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:12'
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
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2L
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
:?????????
 
_user_specified_nameinputs
?
?
__inference_loss_fn_1_14385816M
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
?
?
/__inference_m_2__to__m_3_layer_call_fn_14385410

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
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_143833352
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
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_14383450

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
$serving_default_m_0__to__m_1_input:0?????????B
m_14__to__m_150
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
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
	variables
trainable_variables
regularization_losses
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"??
_tf_keras_sequential??{"name": "L-14-m_0-2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "L-14-m_0-2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "m_0__to__m_1_input"}}, {"class_name": "Dense", "config": {"name": "m_0__to__m_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_1__to__m_2", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_2__to__m_3", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_3__to__m_4", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_4__to__m_5", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_5__to__m_6", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_6__to__m_7", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_7__to__m_8", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_8__to__m_9", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_9__to__m_10", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_10__to__m_11", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_11__to__m_12", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_12__to__m_13", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_13__to__m_14", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "m_14__to__m_15", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "shared_object_id": 61, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 2]}, "float32", "m_0__to__m_1_input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "L-14-m_0-2", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "m_0__to__m_1_input"}, "shared_object_id": 0}, {"class_name": "Dense", "config": {"name": "m_0__to__m_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}, {"class_name": "Dense", "config": {"name": "m_1__to__m_2", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 7}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8}, {"class_name": "Dense", "config": {"name": "m_2__to__m_3", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "m_3__to__m_4", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 15}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16}, {"class_name": "Dense", "config": {"name": "m_4__to__m_5", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 19}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20}, {"class_name": "Dense", "config": {"name": "m_5__to__m_6", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 23}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24}, {"class_name": "Dense", "config": {"name": "m_6__to__m_7", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 27}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28}, {"class_name": "Dense", "config": {"name": "m_7__to__m_8", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32}, {"class_name": "Dense", "config": {"name": "m_8__to__m_9", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 35}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36}, {"class_name": "Dense", "config": {"name": "m_9__to__m_10", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 39}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40}, {"class_name": "Dense", "config": {"name": "m_10__to__m_11", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 43}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 44}, {"class_name": "Dense", "config": {"name": "m_11__to__m_12", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 47}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 48}, {"class_name": "Dense", "config": {"name": "m_12__to__m_13", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 51}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 52}, {"class_name": "Dense", "config": {"name": "m_13__to__m_14", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 55}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 56}, {"class_name": "Dense", "config": {"name": "m_14__to__m_15", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 59}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60}]}}, "training_config": {"loss": "categorical_crossentropy", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0005000000237487257, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?


kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_0__to__m_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_0__to__m_1", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 3}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}, "shared_object_id": 62}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?	

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_1__to__m_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_1__to__m_2", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 7}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 63}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

"kernel
#bias
$	variables
%trainable_variables
&regularization_losses
'	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_2__to__m_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_2__to__m_3", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 11}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 64}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

(kernel
)bias
*	variables
+trainable_variables
,regularization_losses
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_3__to__m_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_3__to__m_4", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 15}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 65}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

.kernel
/bias
0	variables
1trainable_variables
2regularization_losses
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_4__to__m_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_4__to__m_5", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 17}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 18}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 19}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 20, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 66}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

4kernel
5bias
6	variables
7trainable_variables
8regularization_losses
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_5__to__m_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_5__to__m_6", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 21}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 22}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 23}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 24, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 67}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

:kernel
;bias
<	variables
=trainable_variables
>regularization_losses
?	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_6__to__m_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_6__to__m_7", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 25}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 26}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 27}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 68}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

@kernel
Abias
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_7__to__m_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_7__to__m_8", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 29}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 30}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 31}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 32, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 69}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

Fkernel
Gbias
H	variables
Itrainable_variables
Jregularization_losses
K	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_8__to__m_9", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_8__to__m_9", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 33}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 34}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 35}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 70}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

Lkernel
Mbias
N	variables
Otrainable_variables
Pregularization_losses
Q	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_9__to__m_10", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_9__to__m_10", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 39}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 40, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 71}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

Rkernel
Sbias
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_10__to__m_11", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_10__to__m_11", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 43}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 44, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 72}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

Xkernel
Ybias
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_11__to__m_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_11__to__m_12", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 45}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 46}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 47}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 48, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 73}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

^kernel
_bias
`	variables
atrainable_variables
bregularization_losses
c	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_12__to__m_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_12__to__m_13", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 49}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 50}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 51}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 52, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 74}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

dkernel
ebias
f	variables
gtrainable_variables
hregularization_losses
i	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_13__to__m_14", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_13__to__m_14", "trainable": true, "dtype": "float32", "units": 49, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "HeNormal", "config": {"seed": null}, "shared_object_id": 53}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 54}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 55}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 56, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 75}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?	

jkernel
kbias
l	variables
mtrainable_variables
nregularization_losses
o	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "m_14__to__m_15", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "m_14__to__m_15", "trainable": true, "dtype": "float32", "units": 2, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.00019999999494757503}, "shared_object_id": 59}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 60, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 49}}, "shared_object_id": 76}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 49]}}
?
piter

qbeta_1

rbeta_2
	sdecay
tlearning_ratem?m?m?m?"m?#m?(m?)m?.m?/m?4m?5m?:m?;m?@m?Am?Fm?Gm?Lm?Mm?Rm?Sm?Xm?Ym?^m?_m?dm?em?jm?km?v?v?v?v?"v?#v?(v?)v?.v?/v?4v?5v?:v?;v?@v?Av?Fv?Gv?Lv?Mv?Rv?Sv?Xv?Yv?^v?_v?dv?ev?jv?kv?"
	optimizer
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
ulayer_metrics
	variables
trainable_variables
vlayer_regularization_losses
regularization_losses

wlayers
xnon_trainable_variables
ymetrics
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
%:#12m_0__to__m_1/kernel
:12m_0__to__m_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
zlayer_metrics
	variables
trainable_variables
{layer_regularization_losses
regularization_losses

|layers
}non_trainable_variables
~metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_1__to__m_2/kernel
:12m_1__to__m_2/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
layer_metrics
	variables
trainable_variables
 ?layer_regularization_losses
 regularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_2__to__m_3/kernel
:12m_2__to__m_3/bias
.
"0
#1"
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
$	variables
%trainable_variables
 ?layer_regularization_losses
&regularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_3__to__m_4/kernel
:12m_3__to__m_4/bias
.
(0
)1"
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
*	variables
+trainable_variables
 ?layer_regularization_losses
,regularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_4__to__m_5/kernel
:12m_4__to__m_5/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
0	variables
1trainable_variables
 ?layer_regularization_losses
2regularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_5__to__m_6/kernel
:12m_5__to__m_6/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
6	variables
7trainable_variables
 ?layer_regularization_losses
8regularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_6__to__m_7/kernel
:12m_6__to__m_7/bias
.
:0
;1"
trackable_list_wrapper
.
:0
;1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
<	variables
=trainable_variables
 ?layer_regularization_losses
>regularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_7__to__m_8/kernel
:12m_7__to__m_8/bias
.
@0
A1"
trackable_list_wrapper
.
@0
A1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
B	variables
Ctrainable_variables
 ?layer_regularization_losses
Dregularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
%:#112m_8__to__m_9/kernel
:12m_8__to__m_9/bias
.
F0
G1"
trackable_list_wrapper
.
F0
G1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
H	variables
Itrainable_variables
 ?layer_regularization_losses
Jregularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$112m_9__to__m_10/kernel
 :12m_9__to__m_10/bias
.
L0
M1"
trackable_list_wrapper
.
L0
M1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
N	variables
Otrainable_variables
 ?layer_regularization_losses
Pregularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%112m_10__to__m_11/kernel
!:12m_10__to__m_11/bias
.
R0
S1"
trackable_list_wrapper
.
R0
S1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
T	variables
Utrainable_variables
 ?layer_regularization_losses
Vregularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%112m_11__to__m_12/kernel
!:12m_11__to__m_12/bias
.
X0
Y1"
trackable_list_wrapper
.
X0
Y1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
Z	variables
[trainable_variables
 ?layer_regularization_losses
\regularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%112m_12__to__m_13/kernel
!:12m_12__to__m_13/bias
.
^0
_1"
trackable_list_wrapper
.
^0
_1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
`	variables
atrainable_variables
 ?layer_regularization_losses
bregularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%112m_13__to__m_14/kernel
!:12m_13__to__m_14/bias
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
f	variables
gtrainable_variables
 ?layer_regularization_losses
hregularization_losses
?layers
?non_trainable_variables
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
':%12m_14__to__m_15/kernel
!:2m_14__to__m_15/bias
.
j0
k1"
trackable_list_wrapper
.
j0
k1"
trackable_list_wrapper
(
?0"
trackable_list_wrapper
?
?layer_metrics
l	variables
mtrainable_variables
 ?layer_regularization_losses
nregularization_losses
?layers
?non_trainable_variables
?metrics
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
trackable_dict_wrapper
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
trackable_list_wrapper
(
?0"
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
*:(12Adam/m_0__to__m_1/kernel/m
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
,:*12Adam/m_14__to__m_15/kernel/m
&:$2Adam/m_14__to__m_15/bias/m
*:(12Adam/m_0__to__m_1/kernel/v
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
,:*12Adam/m_14__to__m_15/kernel/v
&:$2Adam/m_14__to__m_15/bias/v
?2?
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14384985
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14385184
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14384454
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14384623?
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
#__inference__wrapped_model_14383265?
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
m_0__to__m_1_input?????????
?2?
-__inference_L-14-m_0-2_layer_call_fn_14383771
-__inference_L-14-m_0-2_layer_call_fn_14385249
-__inference_L-14-m_0-2_layer_call_fn_14385314
-__inference_L-14-m_0-2_layer_call_fn_14384285?
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
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_14385337?
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
/__inference_m_0__to__m_1_layer_call_fn_14385346?
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
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_14385369?
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
/__inference_m_1__to__m_2_layer_call_fn_14385378?
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
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_14385401?
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
/__inference_m_2__to__m_3_layer_call_fn_14385410?
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
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_14385433?
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
/__inference_m_3__to__m_4_layer_call_fn_14385442?
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
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_14385465?
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
/__inference_m_4__to__m_5_layer_call_fn_14385474?
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
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_14385497?
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
/__inference_m_5__to__m_6_layer_call_fn_14385506?
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
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_14385529?
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
/__inference_m_6__to__m_7_layer_call_fn_14385538?
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
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_14385561?
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
/__inference_m_7__to__m_8_layer_call_fn_14385570?
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
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_14385593?
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
/__inference_m_8__to__m_9_layer_call_fn_14385602?
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
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_14385625?
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
0__inference_m_9__to__m_10_layer_call_fn_14385634?
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
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_14385657?
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
1__inference_m_10__to__m_11_layer_call_fn_14385666?
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
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_14385689?
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
1__inference_m_11__to__m_12_layer_call_fn_14385698?
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
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_14385721?
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
1__inference_m_12__to__m_13_layer_call_fn_14385730?
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
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_14385753?
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
1__inference_m_13__to__m_14_layer_call_fn_14385762?
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
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_14385785?
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
1__inference_m_14__to__m_15_layer_call_fn_14385794?
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
__inference_loss_fn_0_14385805?
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
__inference_loss_fn_1_14385816?
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
__inference_loss_fn_2_14385827?
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
__inference_loss_fn_3_14385838?
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
__inference_loss_fn_4_14385849?
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
__inference_loss_fn_5_14385860?
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
__inference_loss_fn_6_14385871?
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
__inference_loss_fn_7_14385882?
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
__inference_loss_fn_8_14385893?
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
__inference_loss_fn_9_14385904?
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
__inference_loss_fn_10_14385915?
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
__inference_loss_fn_11_14385926?
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
__inference_loss_fn_12_14385937?
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
__inference_loss_fn_13_14385948?
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
__inference_loss_fn_14_14385959?
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
&__inference_signature_wrapper_14384786m_0__to__m_1_input"?
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
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14384454?"#()./45:;@AFGLMRSXY^_dejkC?@
9?6
,?)
m_0__to__m_1_input?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14384623?"#()./45:;@AFGLMRSXY^_dejkC?@
9?6
,?)
m_0__to__m_1_input?????????
p

 
? "%?"
?
0?????????
? ?
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14384985?"#()./45:;@AFGLMRSXY^_dejk7?4
-?*
 ?
inputs?????????
p 

 
? "%?"
?
0?????????
? ?
H__inference_L-14-m_0-2_layer_call_and_return_conditional_losses_14385184?"#()./45:;@AFGLMRSXY^_dejk7?4
-?*
 ?
inputs?????????
p

 
? "%?"
?
0?????????
? ?
-__inference_L-14-m_0-2_layer_call_fn_14383771"#()./45:;@AFGLMRSXY^_dejkC?@
9?6
,?)
m_0__to__m_1_input?????????
p 

 
? "???????????
-__inference_L-14-m_0-2_layer_call_fn_14384285"#()./45:;@AFGLMRSXY^_dejkC?@
9?6
,?)
m_0__to__m_1_input?????????
p

 
? "???????????
-__inference_L-14-m_0-2_layer_call_fn_14385249s"#()./45:;@AFGLMRSXY^_dejk7?4
-?*
 ?
inputs?????????
p 

 
? "???????????
-__inference_L-14-m_0-2_layer_call_fn_14385314s"#()./45:;@AFGLMRSXY^_dejk7?4
-?*
 ?
inputs?????????
p

 
? "???????????
#__inference__wrapped_model_14383265?"#()./45:;@AFGLMRSXY^_dejk;?8
1?.
,?)
m_0__to__m_1_input?????????
? "??<
:
m_14__to__m_15(?%
m_14__to__m_15?????????=
__inference_loss_fn_0_14385805?

? 
? "? >
__inference_loss_fn_10_14385915R?

? 
? "? >
__inference_loss_fn_11_14385926X?

? 
? "? >
__inference_loss_fn_12_14385937^?

? 
? "? >
__inference_loss_fn_13_14385948d?

? 
? "? >
__inference_loss_fn_14_14385959j?

? 
? "? =
__inference_loss_fn_1_14385816?

? 
? "? =
__inference_loss_fn_2_14385827"?

? 
? "? =
__inference_loss_fn_3_14385838(?

? 
? "? =
__inference_loss_fn_4_14385849.?

? 
? "? =
__inference_loss_fn_5_143858604?

? 
? "? =
__inference_loss_fn_6_14385871:?

? 
? "? =
__inference_loss_fn_7_14385882@?

? 
? "? =
__inference_loss_fn_8_14385893F?

? 
? "? =
__inference_loss_fn_9_14385904L?

? 
? "? ?
J__inference_m_0__to__m_1_layer_call_and_return_conditional_losses_14385337\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????1
? ?
/__inference_m_0__to__m_1_layer_call_fn_14385346O/?,
%?"
 ?
inputs?????????
? "??????????1?
L__inference_m_10__to__m_11_layer_call_and_return_conditional_losses_14385657\RS/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
1__inference_m_10__to__m_11_layer_call_fn_14385666ORS/?,
%?"
 ?
inputs?????????1
? "??????????1?
L__inference_m_11__to__m_12_layer_call_and_return_conditional_losses_14385689\XY/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
1__inference_m_11__to__m_12_layer_call_fn_14385698OXY/?,
%?"
 ?
inputs?????????1
? "??????????1?
L__inference_m_12__to__m_13_layer_call_and_return_conditional_losses_14385721\^_/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
1__inference_m_12__to__m_13_layer_call_fn_14385730O^_/?,
%?"
 ?
inputs?????????1
? "??????????1?
L__inference_m_13__to__m_14_layer_call_and_return_conditional_losses_14385753\de/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
1__inference_m_13__to__m_14_layer_call_fn_14385762Ode/?,
%?"
 ?
inputs?????????1
? "??????????1?
L__inference_m_14__to__m_15_layer_call_and_return_conditional_losses_14385785\jk/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????
? ?
1__inference_m_14__to__m_15_layer_call_fn_14385794Ojk/?,
%?"
 ?
inputs?????????1
? "???????????
J__inference_m_1__to__m_2_layer_call_and_return_conditional_losses_14385369\/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_1__to__m_2_layer_call_fn_14385378O/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_2__to__m_3_layer_call_and_return_conditional_losses_14385401\"#/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_2__to__m_3_layer_call_fn_14385410O"#/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_3__to__m_4_layer_call_and_return_conditional_losses_14385433\()/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_3__to__m_4_layer_call_fn_14385442O()/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_4__to__m_5_layer_call_and_return_conditional_losses_14385465\.//?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_4__to__m_5_layer_call_fn_14385474O.//?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_5__to__m_6_layer_call_and_return_conditional_losses_14385497\45/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_5__to__m_6_layer_call_fn_14385506O45/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_6__to__m_7_layer_call_and_return_conditional_losses_14385529\:;/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_6__to__m_7_layer_call_fn_14385538O:;/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_7__to__m_8_layer_call_and_return_conditional_losses_14385561\@A/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_7__to__m_8_layer_call_fn_14385570O@A/?,
%?"
 ?
inputs?????????1
? "??????????1?
J__inference_m_8__to__m_9_layer_call_and_return_conditional_losses_14385593\FG/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
/__inference_m_8__to__m_9_layer_call_fn_14385602OFG/?,
%?"
 ?
inputs?????????1
? "??????????1?
K__inference_m_9__to__m_10_layer_call_and_return_conditional_losses_14385625\LM/?,
%?"
 ?
inputs?????????1
? "%?"
?
0?????????1
? ?
0__inference_m_9__to__m_10_layer_call_fn_14385634OLM/?,
%?"
 ?
inputs?????????1
? "??????????1?
&__inference_signature_wrapper_14384786?"#()./45:;@AFGLMRSXY^_dejkQ?N
? 
G?D
B
m_0__to__m_1_input,?)
m_0__to__m_1_input?????????"??<
:
m_14__to__m_15(?%
m_14__to__m_15?????????