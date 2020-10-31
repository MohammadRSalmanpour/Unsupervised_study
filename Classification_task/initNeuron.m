function [C V] = initNeuron(LB, UB)
    V = (UB-LB)./6;
    C = (LB+UB)./2; 
end