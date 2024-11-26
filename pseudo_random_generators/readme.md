                      The Linear Congruential Generator (LCG)

The Linear Congruential Generator (LCG) is one of the oldest and simplest methods
for generating pseudorandom numbers. It operates based on a linear recurrence relation.
Here’s a breakdown of how it works:

The LCG generates a sequence of pseudorandom numbers using the following formula:

 Xₙ₊₁ = (a ⋅ Xₙ + c) (mod m)

Where:

•  Xₙ  is the current state (or seed) of the generator.

•  Xₙ₊₁  is the next state.

•  a  is the multiplier (a constant).

•  c  is the increment (another constant).

•  m  is the modulus (the range of values).

▎Steps of Operation

1. Initialization:

   • Choose values for  a ,  c , and  m . These values should be carefully
    selected to ensure good statistical properties.

   • Set an initial seed value  X₀ .

2. Generating Numbers:

   • To generate a new random number, apply the formula:

     • Compute  Xₙ₊₁  using the previous value  Xₙ .

     • Repeat this process to generate a sequence of numbers.

3. Normalization:

   • The generated numbers  Xₙ  can be normalized to a desired range
    (e.g., [0, 1)) by dividing by  m :

    Random Number = Xₙ/m

▎Example

Let’s say we choose:

•  m = 16

•  a = 5

•  c = 1

• Initial seed  X₀ = 7

Using these values, we can compute the first few numbers:

1. First number:

   •  X₁ = (5 ⋅ 7 + 1) (mod 16) = 36 (mod 16) = 4

2. Second number:

   •  X₂ = (5 ⋅ 4 + 1) (mod 16) = 21 (mod 16) = 5

3. Third number:

   •  X₃ = (5 ⋅ 5 + 1) (mod 16) = 26 (mod 16) = 10

And so on...
