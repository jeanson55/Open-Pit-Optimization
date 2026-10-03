import numpy as np


class PitSolution:

    def __init__(self, chromosome, epv):
        """
        chromosome:
            numpy array of 0/1 block selection

        epv:
            economic values of blocks
        """

        self.x = chromosome.astype(np.int8)
        self.epv = epv


    def objective(self):
        """
        Total economic value
        """

        return np.sum(
            self.epv[self.x == 1]
        )


    def blocks(self):
        """
        Number of selected blocks
        """

        return np.sum(self.x)



def repair_solution(
        chromosome,
        graph
):
    """
    Enforce precedence constraints.

    If a block is selected, every required successor in its outgoing
    precedence arcs must also be selected.
    """

    repaired = chromosome.copy()


    changed = True

    while changed:

        changed = False


        selected = np.where(
            repaired == 1
        )[0]


        for block in selected:

            node = block + 1


            requirements = list(
                graph.successors(node)
            )


            for required in requirements:


                # Ignore the sink and any non-block nodes.

                if required == len(repaired) + 1:
                    continue

                required_block = required - 1


                if 0 <= required_block < len(repaired) and repaired[required_block] == 0:

                    repaired[required_block] = 1
                    changed = True


    return repaired