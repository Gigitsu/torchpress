Impressions with Lua and Torch
======
Introduction to torch and why it is implemented using Lua.
[Small lua tutorial](http://tylerneylon.com/a/learn-lua/)

How to define a Class in torch
------

Classes are not built in Lua, but can be defined in two ways:

1. using the table primitive
2. using the [library class](https://github.com/torch/class).


### 1. Class using Tables

```lua

Dog = {}                                   -- 1.

function Dog:new()                         -- 2.
  newObj = {sound = 'woof'}                -- 3.
  self.__index = self                      -- 4.
  return setmetatable(newObj, self)        -- 5.
end

function Dog:makeSound()                   -- 6.
  print('I say ' .. self.sound)
end

mrDog = Dog:new()                          -- 7.
mrDog:makeSound()  -- 'I say woof'         -- 8.

```
In line 1 it is defined an empty table. Lines 2:5 defines the `new` function. The `setmemtable` at line 5 allows Lua to add an entry in the `Dog` table. We can check this by printing the `Dog` into the torch shell

```
th> Dog
{
  new : function: 0x066a3358
}

```

### 2. Using the library Class

```lua
local DataSource = torch.class('data.Datasource')

function DataSource:__init()
   -- init the class variables
   --self.foo = 'bar'
end

function DataSource:foo()
	return 'bar'
end
```


### Static Methods

Static methods are defined as follows and they cannot access to object variables.

```lua

function DataSource.bar()
	return 'foo'
end

```