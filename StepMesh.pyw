from multiprocessing import freeze_support

from stepmesh.gui.app import main

if __name__ == "__main__":
    freeze_support()  # needed for multi-core meshing inside the packaged exe
    main()
