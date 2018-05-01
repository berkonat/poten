local Force, parent = torch.class('poten.Force', 'poten.Potential')

function Force:__init(potenModule)
   parent = potenModule:clone()
end

function Force:updateOutput(input)
      return input.poten.Force_updateOutput(self, input)
end

