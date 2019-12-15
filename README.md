# final_projects
# Monte Carlo Simulation on Lighting Equation

The only code file written by me is "pathtracing.py", all the other files are cited from https://github.com/fmcs3/PathTracing

An object would collect all the light that would hit it, absorb a certain portion of it, and reflect the rest of it back. Then, our eyes are able to perceive this reflected light as the object’s color. The hypothesis is: “By simulating the behavior of light and objects, can we predict this color with simulation?”

First note that the “collection of all light that can hit an object” is an integration problem. Consider this picture:
![alt text](https://github.com/sophiesummer/final_projects/blob/master/pictures/p1.png " ")

We can see that the light contribution can be modeled as a function over “all possible directions”. In this simplified 2D example, we have black up to a certain angle, then green, then yellow, then blue. Then, to “collect” all the light rays, we simply integrate over “all possible directions”, which can be expressed as a half-circle (or a half-sphere in our 3D world). Note that this is a half-circle, because any light “behind” this point cannot possibly reach this point. Then, to simulate the object “absorbing a certain portion of light and reflecting some of it back”, we modify the integration problem to include the surface’s interaction with light. This is called “the rendering equation”:
![alt text](https://github.com/sophiesummer/final_projects/blob/master/pictures/p2.png " ")

The integration bound is Ω, which refers to “all possible directions”. ωi represents a ray of light from a particular direction. Then, Li(ωi) represents “the amount of light energy of that particular ray”. Next we have c/π - we use c to mean “the color” of our object’s surface, which we manually define for every object, which is always less than 1. So, by multiplying our incoming light energy by the “color” of our object (which, being less than 1, “takes away” from the total value), we can model the object reflecting only a portion of light back to the world (this model is called the “Lambertian BDRF”). The last portion, (ωi • n), models the difference between the light’s incoming direction and the surface “normal” - the direction that our surface is facing.
![alt text](https://github.com/sophiesummer/final_projects/blob/master/pictures/p3.png " ")

Consider this sphere. Notice that as the surface of the sphere “turns away” from the light, the difference between that light’s direction and the surface grows - and consequently that light has less influence over that surface. That’s what the (ωi • n) term models. Lastly, we note that any object can “give off” any amount of light simply by having a lot of stored up energy.

### Implementation: Monte Carlo and Rejection sampling.
the code to generate uniform ray directions using rejection sampling:
![alt text](https://github.com/sophiesummer/final_projects/blob/master/pictures/p10.png " ")

We can estimate this integral with Monte-Carlo methods. Here is a direct translation:
![alt text](https://github.com/sophiesummer/final_projects/blob/master/pictures/p8.png " ")
The added 2π is a normalizing constant for the integration bound - a 3D half-sphere N has a surface area of exactly 2π. We now need to generate samples of uniformly-distributed ray directions. We do this with rejection sampling. We start by generating a 3-D coordinate sample (x, y, z) over the unit half-box using 3 uniform samples between [−1,1], [−1,1], and [0,1] respectively. Then, we test if that point is inside the unit hemisphere by comparing x2 + y2 + z2 ≤ 1. If it is, we accept this point. To get a ray direction from this point, we simply create a ray from the origin to our uniformly distributed point. Here is an example illustration in simpler 2D:
![alt text](https://github.com/sophiesummer/final_projects/blob/master/pictures/p4.png " ")
Since the points are uniformly distributed inside a sphere, its direction from the origin must also uniformly distributed.

To collect a sample “path”, we have this recursive function:
![alt text](https://github.com/sophiesummer/final_projects/blob/master/pictures/p9.png " ")

The code to generate our desired SPP sample paths:
![alt text](https://github.com/sophiesummer/final_projects/blob/master/pictures/p11.png " ")

The code to generate uniform ray directions using rejection sampling:
![alt text](https://github.com/sophiesummer/final_projects/blob/master/pictures/p10.png " ")


#### Conclusion
We can see that just by defining some parameters of our world that we want to visualize (namely only the surface normal and the color for each object in the world) we can accurately predict the color of our world at any given point, which leads to a completely simulated image.
