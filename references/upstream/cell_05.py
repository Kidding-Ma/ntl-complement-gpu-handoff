# complementary public stack in its own process (frees all GPU memory when done); its wall time is charged to the ATLAS budget
import time as _t
_c0 = _t.time()
import os; os.environ['NTL_DROP_Q35'] = '1'
!python comp_stack.py 2>&1 | grep -vE 'Warning|warn'
COMP_SEC = _t.time() - _c0
print('comp stack wall time', round(COMP_SEC))
