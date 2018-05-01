local Energy, parent = torch.class('poten.Energy', 'poten.Potential')

function Energy:__init(potenParams,inputStructure)
   parent.__init(self,potenParams,inputStructure)
end

function Energy:set(numParts, unitCell, partList, partSpecies, atomPositions)
   self.numParts = numParts or self.numParts
   self.unitCell = unitCell or self.unitCell
   self.partList = partList or self.partList
   self.partSpecies = partSpecies or self.partSpecies
   self.atomPositions = atomPositions or self.atomPositions
   self.inputStructure = {self.numParts,self.unitCell,self.partList,self.partSpecies,self.atomPositions}
end

function Energy:update(input)
--   if input then
--      return self.atomPositions.poten.Energy_updateList(self, input)
--   else
      return self.output.poten.Energy_updateAll(self)
--   end
end

function Energy:neighborUpdate(input)
--   if input then
--      return self.atomPositions.poten.Energy_updateList(self, input)
--   else
      return self.outputList.poten.Energy_neighborUpdateAll(self)
--   end
end

function Energy:neighborPairUpdate(input)
--   if input then
--      return self.atomPositions.poten.Energy_updateList(self, input)
--   else
      return self.outputList.poten.Energy_neighborPairUpdateAll(self)
--   end
end

function Energy:neighborPairUpdate(input)
--   if input then
--      return self.atomPositions.poten.Energy_updateList(self, input)
--   else
      return self.outputList.poten.Energy_neighborPairUpdateAll(self)
--   end
end

function Energy:inputUpdate(input)
--   if input then
--      return self.atomPositions.poten.Energy_updateList(self, input)
--   else
      return self.output.poten.Energy_inputUpdateAll(self)
--   end
end

function Energy:pairUpdate(input)
--   if input then
--      return self.atomPositions.poten.Energy_updateList(self, input)
--   else
      return self.output.poten.Energy_pairUpdateAll(self)
--   end
end

function Energy:updateForces(input)
      return input.poten.Energy_updateAllForces(self, input)
end

function Energy:pairForces(input)
      return input.poten.Energy_pairAllForces(self, input)
end

function Energy:updateForceContrib(input)
      return input.poten.Energy_updateAllForceContributions(self, input)
end

function Energy:calculateForces(input,runthreaded)
      return input.poten.Energy_calculateAllForces(self, input, runthreaded)
end

function Energy:calculateNNForces(input,runthreaded)
      return input.poten.Energy_calculateAllnnForces(self, input, runthreaded)
end

function Energy:calculatePairForces(input,runthreaded)
      return input.poten.Energy_calculateAllPairForces(self, input, runthreaded)
end

function Energy:updateOutput(input)
   return Energy:update(input)
end

function Energy:setStructure(input)
   self.numParts=input[1]
   self.unitCell=input[2]
   self.partList=input[3]
   self.partSpecies=input[4]
   self.atomPositions=input[5]
   self.inputStructure=input
end

function Energy:setParticles(numParts, partList, partSpecies, atomPositions)
   self:set(numParts, self.unitCell, partList, partSpecies, atomPositions)
end

function Energy:setPositions(partList, atomPositions)
   self:set(self.numParts, self.unitCell, partList, self.partSpecies, atomPositions)
end

