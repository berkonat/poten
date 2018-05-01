local Pair, parent = torch.class('poten.Pair', 'nn.Module')

function Pair:__init()
   parent.__init(self)
   self.pairInput = torch.Tensor() 
end

function Pair:updateOutput(input)
   return input.poten.Pair_updateOutput(self, input)
end

function Pair:updateGradInput(input, gradOutput)
   return input.poten.Pair_updateGradInput(self, input, gradOutput)
end
