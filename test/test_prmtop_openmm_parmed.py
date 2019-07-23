import sys
from pathlib import Path
import pytest

import simtk
from simtk import openmm, unit
from simtk.openmm import app
from mdtraj.utils import enter_temp_directory

import parmed as pmd


def energy(prmtop, inpcrd):
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


def test_prmtop_openmm_parmed():
    inpcrd = "benzene.inpcrd"
    prmtop = "benzene.prmtop"
    openmm_prmtop = app.AmberPrmtopFile(prmtop)
    openmm_system = openmm_prmtop.createSystem(
        nonbondedMethod=app.NoCutoff,
        nonbondedCutoff=99.9 * unit.nanometers,
        constraints=None,
        rigidWater=False,
    )
    parmed_prmtop = pmd.load_file(prmtop)
    parmed_parmset = pmd.amber.AmberParm.from_structure(parmed_prmtop)
    parmed_prmtop_from_openmm = pmd.openmm.load_topology(openmm_prmtop.topology, openmm_system)
    parmed_parmset_from_openmm = pmd.amber.AmberParm.from_structure(parmed_prmtop_from_openmm)
    assert parmed_parmset.dihedral_types == parmed_parmset_from_openmm.dihedral_types
    assert parmed_prmtop.dihedral_types == parmed_prmtop_from_openmm.dihedral_types
    parmed_prmtop_from_openmm.save("openmm_to_parmed.prmtop", overwrite=True)
    assert energy(prmtop, inpcrd) == energy("openmm_to_parmed.prmtop", inpcrd)
