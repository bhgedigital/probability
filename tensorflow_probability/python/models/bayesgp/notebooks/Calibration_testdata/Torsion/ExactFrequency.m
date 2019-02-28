function val = ExactFrequency(J1,J2,K1,K2,K3)

 A = J1*K2 + J2*K1 + J1*K3 + J2*K2;
    
    D = 2*J1*J2;
    
    B = (J1*(K2+K3))^2 - 2*J1*J2*(K1*K2 + K1*K3 - K2^2 + K2*K3) + (J2*(K1+K2))^2;
    
    C = sqrt(B);
    
    val = sqrt((A/D) - (C/D));