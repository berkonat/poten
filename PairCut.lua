local PairCut, parent = torch.class('poten.PairCut', 'nn.Module')

function PairCut:__init(paircut)
   parent.__init(self)
   self.paircut = paircut or 2.0
   self.pairInput = torch.Tensor() 
   self.pairCutoff = torch.Tensor(1):fill(self.paircut) 
   self.pairFcutij = torch.Tensor() 
end

function PairCut:updateOutput(input)
   self.pairCutoff:resizeAs(input):fill(self.paircut)
   return input.poten.PairCut_updateOutput(self, input)
end

function PairCut:updateGradInput(input, gradOutput)
   return input.poten.PairCut_updateGradInput(self, input, gradOutput)
end
