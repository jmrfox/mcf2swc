from mcf2swc import example_mesh


if __name__ == "__main__":
    cylinder = example_mesh("cylinder")
    cylinder.export("data/mesh/demo/cylinder.obj")
    torus = example_mesh("torus")
    torus.export("data/mesh/demo/torus.obj")