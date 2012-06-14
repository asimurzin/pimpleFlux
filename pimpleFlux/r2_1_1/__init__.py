#!/usr/bin/env python

#--------------------------------------------------------------------------------------
## pythonFlu - Python wrapping for OpenFOAM C++ API
## Copyright (C) 2010- Alexey Petrov
## Copyright (C) 2009-2010 Pebble Bed Modular Reactor (Pty) Limited (PBMR)
## 
## This program is free software: you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation, either version 3 of the License, or
## (at your option) any later version.
##
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.
## 
## See http://sourceforge.net/projects/pythonflu
##
## Author : Alexey PETROV, Andrey Simurzin
##


#---------------------------------------------------------------------------
from Foam import ref, man


#---------------------------------------------------------------------------
def _createFields( runTime, mesh ):
    
    ref.ext_Info() << "Reading field p\n" << ref.nl
    p = man.volScalarField( man.IOobject( ref.word( "p" ),
                                          ref.fileName( runTime.timeName() ),
                                          mesh,
                                          ref.IOobject.MUST_READ,
                                          ref.IOobject.AUTO_WRITE ),
                            mesh )
    
    ref.ext_Info() << "Reading field U\n" << ref.nl

    U = man.volVectorField( man.IOobject( ref.word( "U" ),
                                          ref.fileName( runTime.timeName() ),
                                          mesh,
                                          ref.IOobject.MUST_READ,
                                          ref.IOobject.AUTO_WRITE ),
                            mesh )
  
    phi = man.createPhi( runTime, mesh, U )
    
    pRefCell = 0
    pRefValue = 0.0
    
    pRefCell, pRefValue = ref.setRefCell( p, mesh.solutionDict().subDict( ref.word( "PIMPLE" ) ), pRefCell, pRefValue )
    
    laminarTransport = man.singlePhaseTransportModel( U, phi )
    
    turbulence = man.incompressible.turbulenceModel.New( U, phi, laminarTransport )
    
    sources = man.IObasicSourceList( mesh )

    return p, U, phi, turbulence, pRefCell, pRefValue, laminarTransport, sources


#--------------------------------------------------------------------------------------
def Ueqn( mesh, pimple, phi, U, p, turbulence, sources ):
    # The initial C++ expression does not work properly, because of
    #  1. turbulence.divDevRhoReff( U ) - changes values for the U boundaries
    #  2. the order of expression arguments computation differs with C++
    #UEqn = man.fvm.ddt(U) + man.fvm.div(phi, U) + man.fvVectorMatrix( turbulence.divDevReff( U ), man.Deps( turbulence, U ) )           
    UEqn = man.fvVectorMatrix( turbulence.divDevReff( U ), man.Deps( turbulence, U ) )  + ( man.fvm.ddt(U) + man.fvm.div(phi, U) )
    UEqn.relax()
    sources.constrain( UEqn )
    
    rAU = man.volScalarField( 1.0/UEqn.A(), man.Deps( UEqn ) )

    if pimple.momentumPredictor():
       ref.solve( UEqn == -ref.fvc.grad( p ) + sources( U ) )
       pass
    
    return UEqn, rAU


#--------------------------------------------------------------------------------------
def pEqn( runTime, mesh, pimple, U, rAU, UEqn, phi, p, pRefCell, pRefValue, cumulativeContErr, sources ): 
    U << rAU() * ( UEqn == sources( U ) ).H()
    if ( pimple.nCorrPISO() <= 1 ):
       # UEqn.clear()
       pass
       
    phi << ( ref.fvc.interpolate( U ) & mesh.Sf() ) + ref.fvc.ddtPhiCorr( rAU, U, phi )
 
    ref.adjustPhi( phi, U, p )

    # Non-orthogonal pressure corrector loop
    while pimple.correctNonOrthogonal():
        #Pressure corrector
        pEqn = ref.fvm.laplacian( rAU, p ) == ref.fvc.div( phi )
        pEqn.setReference( pRefCell, pRefValue )
        
        pEqn.solve( mesh.solver( p.select( pimple.finalInnerIter() ) ) )
           
        if pimple.finalNonOrthogonalIter():
           phi -= pEqn.flux()
           pass
        pass
    cumulativeContErr = ref.ContinuityErrs( phi, runTime, mesh, cumulativeContErr )
    
    # Explicitly relax pressure for momentum corrector
    p.relax()

    U -= rAU * ref.fvc.grad( p )
    U.correctBoundaryConditions()
    sources.correct(U)

    return cumulativeContErr


#--------------------------------------------------------------------------------------
def main_standalone( argc, argv ):

    args = ref.setRootCase( argc, argv )

    runTime = man.createTime( args )

    mesh = man.createMesh( runTime )

    p, U, phi, turbulence, pRefCell, pRefValue, laminarTransport, sources = _createFields( runTime, mesh )
    
    cumulativeContErr = ref.initContinuityErrs()
    
    pimple = man.pimpleControl( mesh )
    
    ref.ext_Info() << "\nStarting time loop\n" <<ref.nl
    
    while runTime.run() :
        adjustTimeStep, maxCo, maxDeltaT = ref.readTimeControls( runTime )

        CoNum, meanCoNum = ref.CourantNo( mesh, phi, runTime )
      
        runTime = ref.setDeltaT( runTime, adjustTimeStep, maxCo, maxDeltaT, CoNum )
        
        runTime.increment()
                
        ref.ext_Info() << "Time = " << runTime.timeName() << ref.nl << ref.nl
        
        # --- Pressure-velocity PIMPLE corrector loop
        while pimple.loop():

            UEqn, rAU = Ueqn( mesh, pimple, phi, U, p, turbulence, sources )
            
            # --- Pressure corrector loop
            while pimple.correct():
               cumulativeContErr = pEqn( runTime, mesh, pimple, U, rAU, UEqn, phi, p, pRefCell, pRefValue, cumulativeContErr, sources )
               pass
            
            if pimple.turbCorr():                             
                turbulence.correct()
                pass
            pass

        runTime.write();
        
        ref.ext_Info() << "ExecutionTime = " << runTime.elapsedCpuTime() << " s" << \
              "  ClockTime = " << runTime.elapsedClockTime() << " s" << ref.nl << ref.nl
        
        pass

    ref.ext_Info() << "End\n" << ref.nl 

    import os
    return os.EX_OK


#--------------------------------------------------------------------------------------
from Foam import FOAM_VERSION
if FOAM_VERSION( ">=", "020100" ):
   if __name__ == "__main__" :
      import sys, os
      argv = sys.argv
      os._exit( main_standalone( len( argv ), argv ) )
      pass
   pass
else:
   ref.ext_Info() << "\n\n To use this solver it is necessary to SWIG OpenFOAM-2.1.0 or higher\n"
   pass


#--------------------------------------------------------------------------------------
