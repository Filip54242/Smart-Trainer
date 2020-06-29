from gui import BaseGUI

# cuda = torch.cuda.is_available()
cuda = False
BaseGUI(enable_cuda=cuda)
