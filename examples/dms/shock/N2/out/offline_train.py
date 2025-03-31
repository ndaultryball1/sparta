import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WP = torch.float64
torch.set_default_dtype(torch.float64)

class GatedModel(torch.nn.Module):
    def __init__(self,num_inputs, H,num_outputs ):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.fc2 = torch.nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc3 = torch.nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc5 = torch.nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc6 = torch.nn.Linear(H, num_outputs).type(torch.DoubleTensor)


        self.G1 = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.G2 = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.G3 = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)

    def forward(self, x):
        sig = torch.nn.Sigmoid( )
        H1 = torch.relu(self.fc1(x))
        H2 = torch.relu(self.fc2(H1))

        H3 = torch.relu(self.G1(x)) * H2
        H4 = torch.relu(self.fc3(H3))

        H5 = torch.relu(self.G2(x)) * H4
        H6 = torch.relu(self.fc5(H5))

        H7 = torch.relu(self.G3(x)) * H6
        return( sig(self.fc6(H7)) )
    
class DoubleGatedModel(torch.nn.Module):
    def __init__(self,num_inputs, H,num_outputs ):
        super().__init__()
        
        self.fc1 = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.fc2 = torch.nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc3 = torch.nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc5 = torch.nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc6 = torch.nn.Linear(H, 2).type(torch.DoubleTensor)


        self.G1 = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.G2 = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.G3 = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)

        self.fc1_r = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.fc2_r = torch.nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc3_r = torch.nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc5_r = torch.nn.Linear(H, H).type(torch.DoubleTensor)
        self.fc6_r = torch.nn.Linear(H, 1).type(torch.DoubleTensor)


        self.G1_r = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.G2_r = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)
        self.G3_r = torch.nn.Linear(num_inputs, H).type(torch.DoubleTensor)

    def forward(self, x):
        sig = torch.nn.Sigmoid( )
        H1 = torch.tanh(self.fc1(x))
        H2 = torch.tanh(self.fc2(H1))

        H3 = torch.tanh(self.G1(x)) * H2
        H4 = torch.tanh(self.fc3(H3))

        H5 = torch.tanh(self.G2(x)) * H4
        H6 = torch.tanh(self.fc5(H5))

        H7 = torch.tanh(self.G3(x)) * H6
        chi_R = sig(self.fc6(H7))

        H1_r = torch.tanh(self.fc1_r(x))
        H2_r = torch.tanh(self.fc2_r(H1_r))

        H3_r = torch.tanh(self.G1_r(x)) * H2_r
        H4_r = torch.tanh(self.fc3_r(H3_r))

        H5_r = torch.tanh(self.G2_r(x)) * H4_r
        H6_r = torch.tanh(self.fc5_r(H5_r))

        H7_r = torch.tanh(self.G3_r(x)) * H6_r
        r = sig(self.fc6_r(H7_r))
        return(torch.concat(
                [chi_R, 
                r,
     ], dim=-1,
)  )

data_directory = "gated_deeper_more/binary"

input = []
outs = []
for comm in range(1):
    for timestep in range(0, 4,2):
        input.append( torch.load(f"{data_directory}/input_{comm}_{timestep}",map_location=device) )
        outs.append( torch.load(f"{data_directory}/out_{comm}_{timestep}",map_location=device))

inputs = torch.concat( input)
out = torch.concat(outs)
e_star = inputs[:,0]
b_star= inputs[:,1]
erot1= inputs[:,2]
erot2= inputs[:,3]

chi = out[:,0]
r  = out[:,1]
try:
    R = out[:,2]
except IndexError:
    print("Only two outputs found")

del input
del outs

model = GatedModel(12,200,3)

epochs = 1000
batch_size=250

LR = 1e-3
optimizer = torch.optim.RMSprop( model.parameters(), lr=LR )

N = len(inputs)

for j in range( epochs ):

    A = 100
    B = 100
    C = 1

    for g in optimizer.param_groups:
        g['lr'] = LR * A / ( B + C * j ) 
    with torch.enable_grad():
        idx =torch.randperm( N )
        train_data = inputs[idx,:]
        train_target=out[idx,:]
        for p in range( 0, N, batch_size ):
            batch = train_data[p:p+batch_size]
            target = train_target[p:p+batch_size]
            pred = model( batch )
            loss = torch.mean( torch.square( pred - target ) ) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    with torch.no_grad():
        pred = model( inputs )
        chi_pred = pred[:,0]
        chi_loss = (chi_pred - chi).square().mean()
        r_pred  = pred[:,1]
        r_loss = (r_pred - r).square().mean()
        try:
            R_pred = out[:,2]
            R_loss = (R_pred - R).square().mean()
        except IndexError:
            print("Only two outputs found")
        train_loss = ( pred- out ).square().mean()
        printstring = f"{j}, {N}, {train_loss:.5e}, {chi_loss:.5e}, {r_loss:.5e}, {R_loss:.5e}, {LR * A / ( B + C * j ) :.5e}\n"
        print(printstring)
        torch.save( model.state_dict(), f"offline/model_N2_{j}.pt" )