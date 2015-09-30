Tips & Tricks
==================
**Abstract:** In this post I will collect some useful lua tips and libraries.

##Method arguments
###Default values
Suppose we have a function with an optional last argument that has a dafault value:
```lua
function foo(arg1, arg2, optionalWithDefault) ... end
```
Lua has no built-in mechanism to handle defautls value for parameters but we can use an expression to simulate this.
Suppose the default value of `optionalWithDefault` is `1`, we can use the following code:
```lua
function foo(arg1, arg2, optionalWithDefault)
  optionalWithDefault = optionalWithDefault or 1
  ...
end
```

Here if `optionalWithDefault` is not specified in `foo` call then `optionalWithDefault` is `nil` and the expression `optionalWithDefault or 1` returns `1` otherwise returns the value of `optionalWithDefault`.
For details on how expression works see [here](http://lua-users.org/wiki/ExpressionsTutorial)

###Named arguments
By default the parameter passing mechanism in Lua is *positional*: when we call a function, arguments match parameters by their positions.
Sometimes, however, it is useful to specify the arguments by name. To illustrate this point, let us consider a function with multiple optional arguments:
```lua
function boo(opt1, opt2, opt3)
  opt1 = opt1 or 1
  opt2 = opt2 or 2
  opt3 = opt3 or 3
  ...
end
```
Either `opt1`, `opt2` and `opt3` have defaults (respectively `1`, `2` and `3`). If we have to call function `boo` but we need a different value only for `opt3` we have two possibilities:
```lua
boo(1, 2, 5) -- we specify all 3 parameters, to do this we must know the default value of opt1 and opt2

boo(nil, nil, 5) -- we can just use nil for opt1 and opt2 but this method can be confusing for longer argument list
```

A third way is to pack all arguments into a table and use that table as the only argument to the function. 
```lua
function boo(args)
  local opt1 = args.opt1 or 1
  local opt2 = args.opt2 or 2
  local opt3 = args.opt3 or 3
  ...
end
```
The special syntax that Lua provides for function calls, with just one table constructor as argument, helps the trick:
```lua
boo{opt3 = 5}
```
Notice the curly brackets instead of simple brackets.

###Improvement
The library [xlua](https://github.com/torch/xlua) ( `$ luarocks install xlua` ) provide a nice support for both named arguments and default values.
Let's use `xlua.unpack` to improve `boo` function:
```lua
function boo(args) -- args is a table
  local localArgs, opt1, opt2, opt3 = xlua.unpack(
    {args},
    'Boo', -- the name of function
    'Testing Boo function', -- a short description
    {arg='opt1', type='number', default=1,
     help='the first optional argument.'},
    {arg='opt2', type='number', default=2,
     help='the second optional argument.'},
    {arg='opt3', type='number', default=3,
     help='the third optional argument.'}
   )
  ...
end
```
Here `xlua.unpack` automatically check the args table, extracting signle argument (`opt1`, `opt2` and `opt3`) and replacing missing arguments with defaults.

##Strings
The [pl.stringx](http://stevedonovan.github.io/Penlight/api/libraries/pl.stringx.html) library provide some useful methods to manipulate strings that lua is missing.

##Split
Split a string into a list of strings using a delimiter.

###Parameters:
 - *self* (string) the string
 - *re* (string) a delimiter (defaults to whitespace) (optional)
 - *n* (int) maximum number of results
 
###Usage:
```lua
#(('one two'):split()) == 2
```
```lua
('one,two,three'):split(',') == List{'one','two','three'}
```
```lua
('one,two,three'):split(',') == List{'one','two','three'}
```
##Splitlines
Break string into a list of lines

###Parameters:
 - *self* (string) the string
 - *keepends* (currently not used)

###Usage:
```lua
txt = [[line1
line2
line3]]

txt:splitlines() == List{'line1','line2','line3'}
