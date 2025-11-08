Heading level 2 (Section)
==========================

.. admonition:: Overview
   :class: Overview

    * **Tutorial:** 10 min

        **Objectives:**
            #. Learn the how Numba works.


=============================
Fortran 95: A Gentle Introduction
=============================

.. contents::
   :local:
   :depth: 2

Purpose
=======

This document is a short introduction of Fortran 95 for scientific and engineering computing.
We will write small programs, learn array-focused features, and compile with common compilers.

What you will learn
-------------------

- Program structure, source layout, and compilation
- Variables, types, and the importance of ``implicit none``
- Control flow: ``if``, ``select case``, ``do`` loops
- Arrays, slicing, intrinsic procedures, and whole-array operations
- Procedures: functions vs subroutines, arguments and ``intent``
- Modules and interfaces
- Dynamic memory with allocatable arrays
- Derived types and simple modular design
- Good practices and common pitfalls in F95

Prerequisites and setup
=======================

- Comfortable with the command line
- Basic programming experience (any language)
- A Fortran compilers on Gadi:
  - GNU Fortran: ``gfortran`` through module ``gcc``
  - Intel Fortran Classic: ``ifort`` through old intel-compiler/* modules
  - Intel Fortran ``ifx`` through intel-compiler-llvm/* modules


Check installation::

  gfortran --version
  # or
  ifort -V
  # or
  ifx -V

Compile and run
---------------

Minimal commands::

  gfortran -Wall -Wextra -O2 hello.f90 -o hello
  ./hello

On systems with multiple compilers, prefer explicit commands and flags.
Use the ``.f90`` extension for all modern Fortran source files (90/95//2003/2008).

Hello, Fortran
==============

Create ``hello.f90``:

.. code-block:: fortran

   program hello
     implicit none
     print *, "Hello, Fortran 95!"
   end program hello

Key idea: ``implicit none`` disables implicit typing and catches many bugs at compile time. Without it, the dedault sets undeclared variables to type ``real`` or ``integer`` based on their names. 

Language basics
===============

Source form
-----------

- Free form source (``.f90``) is standard in F95.
- Comments start with ``!``.
- Indentation is for humans, not the compiler, but keep it consistent.

Types and declarations
----------------------

Built-in numeric and logical types:

.. code-block:: fortran

   integer           :: i, n
   real              :: x, y
   double precision  :: z
   logical           :: convergence
   character(len=20) :: name

Use ``parameter`` for constants (much like ``const`` in C/C++):

.. code-block:: fortran

   integer, parameter :: dp = kind(1.0d0) ! define dp as double precision kind
   real(kind=dp), parameter :: pi = 3.1415926535_dp ! _dp enforces the dp kind

Operators (overview)
--------------------

- Arithmetic: ``+ - * / **``
- Relational: ``== /= < <= > >=``
- Logical: ``.and. .or. .not.``

Input and output
----------------

List-directed I/O (simple and flexible):

.. code-block:: fortran

   integer :: n
   print *, "Enter an integer:"
   read  *, n
   print *, "You entered:", n

Formatted I/O (controlled layout):

.. code-block:: fortran

   real :: x
   x = 12.345
   print '(A, F8.3)', "Value = ", x ! A for string, F8.3 for field width 8 with 3 decimals

Control flow
============

If and case
-----------

.. code-block:: fortran

   integer :: n
   read *, n
   if (n < 0) then
     print *, "negative"
   else if (n == 0) then
     print *, "zero"
   else
     print *, "positive"
   end if

.. code-block:: fortran

   character(len=1) :: c
   read *, c
   select case (c)
   case ('y','Y')
     print *, "yes"
   case ('n','N')
     print *, "no"
   case default
     print *, "unknown"
   end select

Loops
-----

.. code-block:: fortran

   integer :: i, sum
   sum = 0
   do i = 1, 10
     sum = sum + i
   end do
   print *, "Sum 1..10 =", sum

Use ``exit`` to break, ``cycle`` to continue:

.. code-block:: fortran

   integer :: i
   do i = 1, 1000
     if (i*i > 200) exit
     if (mod(i, 2) == 0) cycle
     print *, i
   end do

Arrays and array features
=========================

Declaring arrays
----------------

.. code-block:: fortran

   real :: a(5)              ! fixed-size
   integer :: m, n
   parameter (m=3, n=4)
   real :: b(m, n)           ! 2D array, column-major

Array constructors and assignments
----------------------------------

.. code-block:: fortran

   real :: x(5)
   x = (/ 1.0, 2.0, 3.0, 4.0, 5.0 /)
   x = x * 2.0                ! whole-array operation
   print *, x

Slicing and sections
--------------------

.. code-block:: fortran

   real :: v(10)
   integer :: i
   do i = 1, 10
     v(i) = real(i)
   end do
   print *, v(3:7)            ! subarray 3..7
   print *, v(::2)            ! stride 2

Intrinsic procedures (selected)
-------------------------------

.. code-block:: fortran

   real :: a(4), s
   a = (/1.0, -2.0, 3.0, -4.0/)
   s = sum(a)                 !  -2.0
   print *, minval(a), maxval(a), sum(abs(a))

Element-wise masks and WHERE
----------------------------

.. code-block:: fortran

   real :: t(5)
   t = (/ -1., 0., 1., 2., -2. /)
   where (t < 0.0)
     t = 0.0
   elsewhere
     t = sqrt(t)
   end where
   print *, t

FORALL (F95)
------------

``forall`` expresses element-wise independent assignments. Prefer regular array syntax when possible.

.. code-block:: fortran

   integer, parameter :: n = 5
   real :: iarr(n), oarr(n)
   iarr = (/ (real(i), i=1,n) /)
   forall (i=1:n) oarr(i) = iarr(i) ** 2
   print *, oarr

Procedures
==========

Subroutines vs functions
------------------------

- Functions return a value and are used in expressions.
- Subroutines return results via arguments.

.. code-block:: fortran

   function hypot(a, b) result(h)
     implicit none
     real, intent(in) :: a, b
     real :: h
     h = sqrt(a*a + b*b)
   end function hypot

.. code-block:: fortran

   subroutine normalize(v)
     implicit none
     real, intent(inout) :: v(:)
     real :: s
     s = sqrt(sum(v*v))
     if (s > 0.0) v = v / s
   end subroutine normalize

Intent and optional arguments
-----------------------------

.. code-block:: fortran

   subroutine scale(v, alpha)
     implicit none
     real, intent(inout) :: v(:)
     real, intent(in), optional :: alpha
     real :: k
     k = 1.0
     if (present(alpha)) k = alpha
     v = k * v
   end subroutine scale

Interfaces and explicit interfaces
----------------------------------

An explicit interface is required for assumed-shape arrays and optional arguments.
Modules (next section) provide interfaces automatically.

Modules and program structure
=============================

Modules
-------

Use modules to package procedures, types, and constants.

.. code-block:: fortran

   module linalg
     implicit none
   contains
     subroutine axpy(alpha, x, y)
       real, intent(in)    :: alpha
       real, intent(in)    :: x(:)
       real, intent(inout) :: y(:)
       y = alpha * x + y
     end subroutine axpy
   end module linalg

Use a module with ``use``:

.. code-block:: fortran

   program demo_axpy
     use linalg
     implicit none
     real :: x(3), y(3)
     x = (/1., 2., 3./)
     y = (/4., 5., 6./)
     call axpy(2.0, x, y)
     print *, y
   end program demo_axpy

Access control with ``public`` and ``private``:

.. code-block:: fortran

   module constants
     implicit none
     private
     public :: dp, pi
     integer, parameter :: dp = kind(1.0d0)
     real(kind=dp), parameter :: pi = 3.141592653589793_dp
   end module constants

Dynamic memory with allocatable arrays
======================================

Allocatable arrays
------------------

.. code-block:: fortran

   program dyn
     implicit none
     integer :: n
     real, allocatable :: a(:)

     n = 1000
     allocate(a(n))
     a = 0.0
     a(1) = 1.0
     print *, "size:", size(a)

     deallocate(a)
   end program dyn

Assumed-shape dummy arrays
--------------------------

.. code-block:: fortran

   subroutine center(x)
     implicit none
     real, intent(inout) :: x(:)
     x = x - sum(x)/real(size(x))
   end subroutine center

Derived types
=============

Define simple structures for clarity:

.. code-block:: fortran

   module geom
     implicit none
     type :: point
       real :: x, y
     end type point
   contains
     function distance(a, b) result(d)
       type(point), intent(in) :: a, b
       real :: d
       d = sqrt( (a%x - b%x)**2 + (a%y - b%y)**2 )
     end function distance
   end module geom

Pointers vs allocatables
------------------------

Fortran 95 offers both. Prefer allocatables for dynamic arrays unless you need pointer aliasing.

Recursion, purity, and elemental procedures
===========================================

Enable recursion explicitly in F95:

.. code-block:: fortran

   recursive function fib(n) result(f)
     implicit none
     integer, intent(in) :: n
     integer :: f
     if (n <= 1) then
       f = n
     else
       f = fib(n-1) + fib(n-2)
     end if
   end function fib

Pure procedures promise no side effects on inputs:

.. code-block:: fortran

   pure function sq(x) result(y)
     real, intent(in) :: x
     real :: y
     y = x*x
   end function sq

Elemental procedures apply element-wise to arrays:

.. code-block:: fortran

   elemental function cube(x) result(y)
     real, intent(in) :: x
     real :: y
     y = x*x*x
   end function cube

   ! Usage:
   ! real :: v(3) ; v = (/1.,2.,3./)
   ! print *, cube(v)   ! yields (/1.,8.,27./)

Numerical robustness tips
=========================

- Always use ``implicit none`` in every program unit
- Be explicit about kinds for real numbers when needed
- Use array syntax instead of manual loops when it improves clarity
- Avoid uninitialized variables; compile with warnings
- Validate I/O return codes for robust programs
- Prefer allocatables over pointers for ownership and performance

Project layout and simple builds
================================

A tiny layout::

  src/
    main.f90
    linalg.f90
  build/

Compile with ``gfortran``::

  gfortran -c src/linalg.f90 -o build/linalg.o
  gfortran -c src/main.f90  -o build/main.o
  gfortran build/main.o build/linalg.o -o build/app

Minimal Makefile (optional)::

  FC      = gfortran
  FFLAGS  = -O2 -Wall -Wextra
  OBJ     = build/linalg.o build/main.o

  build/%.o: src/%.f90
  	@mkdir -p build
  	$(FC) $(FFLAGS) -c $< -o $@

  build/app: $(OBJ)
  	$(FC) $(FFLAGS) $^ -o $@

  .PHONY: clean
  clean:
  	rm -rf build

Hands-on exercises
==================

Exercise 1: vector norms
------------------------

Write a module ``vec`` with

- ``function dot(x, y)`` returning the dot product
- ``function norm2(x)`` returning the Euclidean norm

Test with random vectors and verify ``norm2(x)**2 == dot(x, x)`` within a tolerance.

Starter:

.. code-block:: fortran

   module vec
     implicit none
   contains
     function dot(x, y) result(s)
       real, intent(in) :: x(:), y(:)
       real :: s
       s = sum(x*y)
     end function dot

     function norm2(x) result(n)
       real, intent(in) :: x(:)
       real :: n
       n = sqrt(sum(x*x))
     end function norm2
   end module vec

Exercise 2: centered moving average
-----------------------------------

Given a real vector ``x``, compute a centered moving average of window width ``w`` (odd).
Handle boundaries by leaving endpoints unchanged.

Hints:

- Use slicing and ``sum`` over sections
- Consider a separate subroutine ``movavg(x, w, y)`` with ``intent(in)``, ``intent(out)``

Exercise 3: 2D Laplace stencil
------------------------------

Allocate a 2D grid ``u(nx, ny)``, set boundary values, and relax the interior using the five-point stencil
for a fixed number of iterations. Print the maximum change each iteration and stop early if it drops below a tolerance.

Skeleton:

.. code-block:: fortran

   program laplace2d
     implicit none
     integer, parameter :: nx=50, ny=50, itmax=1000
     real, parameter :: tol = 1.0e-5
     real, allocatable :: u(:,:), unew(:,:)
     integer :: it
     real :: err

     allocate(u(nx,ny), unew(nx,ny))
     u = 0.0
     u(:,ny) = 1.0   ! top boundary to 1

     do it = 1, itmax
       unew(2:nx-1,2:ny-1) = 0.25 * ( u(3:nx,2:ny-1) + u(1:nx-2,2:ny-1) &
                                    + u(2:nx-1,3:ny  ) + u(2:nx-1,1:ny-2) )
       unew(:,1)   = u(:,1)
       unew(:,ny)  = u(:,ny)
       unew(1,:)   = u(1,:)
       unew(nx,:)  = u(nx,:)
       err = maxval(abs(unew - u))
       u = unew
       if (err < tol) exit
     end do

     print *, "Iterations:", it, "max change:", err
     deallocate(u, unew)
   end program laplace2d

Common pitfalls in F95
======================

- Forgetting ``implicit none`` leads to hard-to-find bugs
- Mismatched array shapes in assignments or arguments
- Missing explicit interfaces when using assumed-shape arrays
- Using pointers where allocatables suffice
- Overusing ``save`` variables (can hide stateful bugs)
- Off-by-one indices; arrays default to 1-based indexing

Further study
=============

- File I/O with ``open``/``close`` and units
- User-defined operators (beyond scope here)
- Performance basics: array ordering, stride, and temporary arrays
- Multi-file organization and module dependencies

Appendix: compiler flags
========================

Suggested flags for development with GNU Fortran::

  gfortran -std=f95 -Wall -Wextra -Wimplicit-interface -fcheck=all -O2 file.f90 -o app

Strip checks and enable higher optimization for release builds only after thorough testing.


Heading level 2 (Section)
==========================

Heading level 3 (Subsection)
----------------------------

Heading level 4 (Sub-subsection)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Heading level 5 (Paragraph)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Heading level 6 (Subparagraph)
+++++++++++++++++++++++++++++++

Heading level 7 (Lowest level)


Add Imgaes
-----------------

.. image:: ../figs/performance.png


Bullets
---------------------------

 
1. **Annotation and Compilation**: When you use Numba's `@jit` decorator on a Python function, Numba 
first analyzes the function's code. This analysis determines how to compile the function to improve performance. 
You can also provide type hints to help Numba generate more efficient machine code.

2. **Type Inference**: Numba performs type inference on the function’s inputs and outputs. It determines the 
types of variables and ensures that operations are optimized for those types. For example, it might optimize
arithmetic operations for specific numerical types.

3. **Machine Code Generation**: Based on the type information and analysis, Numba generates machine code 
tailored to the function. This code is designed to run directly on the hardware, bypassing the overhead of the 
Python interpreter.

Code Blocks
--------------

..  code-block:: python
    :linenos:

    import numba
    from numba import jit, int32, prange, vectorize, float64, cuda


Notes
--------------

.. note::
 1.  python3/3.11.0
 2.  papi/7.0.1
 3.  openmpi/4.0.1
 4.  cuda/12.3.2
 5.  gcc/14.2.0

Explanations
---------------

.. admonition:: Explanation
   :class: attention
   
    #. Numba is a JIT compiler that optimizes Python code for performance.
    #. It compiles functions at runtime, allowing for efficient execution of numerical computations.
    #. The `@jit` decorator is used to mark functions for optimization.
    #. Numba can handle different input types and adapt its compilation accordingly.


Importance
---------------

.. important::
   In practice weight updates do not happen after  every individual sample; instead, they occur after each batch of data, depending on the **batch size** used. 

Exercise
---------------

.. admonition:: Exercise
   :class: todo

    1. Examine the program *src/distributed_data_parallel.py*. What the changes from data_parallel.ipynb?
    2. Examine the job script *job_scripts/distributed_data_parallel.pbs*.
    3. Run the program using the job script *job_scripts/distributed_data_parallel.pbs*.



.. admonition:: Key Points
   :class: hint

    #. Numba uses simple annonations to parallelise code.
