local Potential = torch.class('poten.Potential')

function Potential:__init(potenParams,inputStructure)
   self.output = torch.Tensor()
   self.outputList = torch.Tensor()
   self.bohrtoang = (1.88*1.88)/10000.0
   self.potenParams = self.potenParams or potenParams
   self.inputStructure = self.inputStructure or inputStructure

   if self.potenParams then 
      self.potCutoff = self.potenParams[1] or 6.0
      self.numTypes = self.potenParams[2] or 2 
      self.numRadial = self.potenParams[3] or 8 
      self.numAngular = self.potenParams[4] or 22 
      self.numSyms = 82
      self.neighborBin = 0.0
      self.etaRad = torch.Tensor(self.numRadial):copy(self.potenParams[5]):mul(self.bohrtoang) 
      self.etaAng = torch.Tensor(self.numAngular):copy(self.potenParams[6]):mul(self.bohrtoang)
      self.lambdaAng = torch.Tensor(self.numAngular):copy(self.potenParams[7])
      self.zetaAng = torch.Tensor(self.numAngular):copy(self.potenParams[8])
      if table.getn(self.potenParams)>8 then
         self.potPairCutoffs = self.potenParams[9]
      else
         self.potPairCutoffs = torch.Tensor(self.numTypes*self.numTypes):zero()
      end
   else
      self.potCutoff = 6.0
      self.numTypes = 2 
      self.numRadial = 8 
      self.numAngular = 22 
      self.numSyms = 82
      self.neighborBin = 0.0
      self.etaRad = torch.Tensor(self.numRadial):mul(self.bohrtoang)
      self.etaAng = torch.Tensor(self.numAngular):mul(self.bohrtoang)
      self.lambdaAng = torch.Tensor(self.numAngular)
      self.zetaAng = torch.Tensor(self.numAngular)
      self.potPairCutoffs = torch.Tensor(self.numTypes*self.numTypes):zero()
   end

   if self.inputStructure then
      self.numParts = self.inputStructure[1]
      self.unitCell = torch.Tensor(3,3):copy(self.inputStructure[2])
      self.partList = torch.Tensor(self.numParts):copy(self.inputStructure[3])
      self.partSpecies = torch.Tensor(self.numParts):copy(self.inputStructure[4])
      self.atomPositions = torch.Tensor(self.numParts,3):copy(self.inputStructure[5])
      self.numNeighList = torch.Tensor(self.numParts)
      self.partNeighList = torch.Tensor(self.numParts,2*self.numParts)
      self.RijList = torch.Tensor(self.numParts,2*self.numParts,3)
   else
      self.numParts = 1
      self.unitCell = torch.Tensor(3,3):zero()
      self.partList = torch.Tensor(self.numParts):fill(1)
      self.partSpecies = torch.Tensor(self.numParts):fill(1)
      self.atomPositions = torch.Tensor(self.numParts,3):zero()
      self.numNeighList = torch.Tensor(self.numParts)
      self.partNeighList = torch.Tensor(self.numParts,2*self.numParts)
      self.RijList = torch.Tensor(self.numParts,2*self.numParts,3)
   end
end

function Potential:updateOutput(input)
   return self.output
end

function Potential:forward(input)
   return self:updateOutput(input)
end

function Potential:clone()
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   return clone
end

function Potential:type(type)
   -- find all tensors and convert them
   for key,param in pairs(self) do
      if torch.typename(param) and torch.typename(param):find('torch%..+Tensor') then
         self[key] = param:type(type)
      end
   end
   return self
end

function Potential:float()
   return self:type('torch.FloatTensor')
end

function Potential:double()
   return self:type('torch.DoubleTensor')
end

function Potential:cuda()
   return self:type('torch.CudaTensor')
end

function Potential:__call__(input)
   self:forward(input)
   return self.output
end
