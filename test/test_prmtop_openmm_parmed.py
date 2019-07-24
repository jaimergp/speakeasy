from pathlib import Path
import pytest
import numpy as np

import simtk
from simtk import openmm, unit
from simtk.openmm import app
from mdtraj.utils import enter_temp_directory
import sander
import parmed as pmd


def datapath(path):
    return str(Path(__file__).parent / path)


def openmm_energy(prmtop, inpcrd):
    prmtop = app.AmberPrmtopFile(prmtop)
    inpcrd = app.AmberInpcrdFile(inpcrd)

    system = prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        nonbondedCutoff=99.9 * unit.nanometers,
        constraints=None,
        rigidWater=False,
    )
    context = openmm.Context(
        system,
        openmm.VerletIntegrator(1 * unit.femtoseconds),
        openmm.Platform.getPlatformByName("CPU"),
    )
    context.setPositions(inpcrd.positions)
    energy = (
        context.getState(getEnergy=True)
        .getPotentialEnergy()
        .value_in_unit(unit.kilocalories_per_mole)
    )
    return energy


def sander_energy(prmtop, inpcrd):
    config = sander.gas_input()
    with sander.setup(prmtop, inpcrd, None, config):
        energy, forces = sander.energy_forces()
    return energy.tot


def test_prmtop_openmm_parmed():
    inpcrd = datapath("benzene.inpcrd")
    prmtop = datapath("benzene.prmtop")
    # Load PRMTOP with OpenMM
    openmm_prmtop = app.AmberPrmtopFile(prmtop)
    openmm_system = openmm_prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        nonbondedCutoff=99.9 * unit.nanometers,
        constraints=None,
        rigidWater=False,
    )
    # Load PRMTOP with ParmEd
    parmed_prmtop = pmd.load_file(prmtop)
    # must be converted to AmberParm to fix dihedrals
    parmed_parmset = pmd.amber.AmberParm.from_structure(parmed_prmtop)
    # Convert to ParmEd Structure from OpenMM topology... *should* be the same
    parmed_prmtop_from_openmm = pmd.openmm.load_topology(openmm_prmtop.topology, openmm_system)
    parmed_parmset_from_openmm = pmd.amber.AmberParm.from_structure(parmed_prmtop_from_openmm)
    parmed_prmtop_from_openmm.save("openmm_to_parmed.prmtop", overwrite=True)

    assert np.isclose(openmm_energy(prmtop, inpcrd), openmm_energy("openmm_to_parmed.prmtop", inpcrd))
    assert np.isclose(sander_energy(prmtop, inpcrd), sander_energy("openmm_to_parmed.prmtop", inpcrd))
    # Energies seem to be the same, but the dihedral types are different
    # Scaling factors are not recovered adequately
    # The OpenMM version has more types inferred too - maybe due to the exclusion system?
    assert parmed_parmset.dihedral_types == parmed_parmset_from_openmm.dihedral_types
