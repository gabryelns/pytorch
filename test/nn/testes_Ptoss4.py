import torch
import unittest
from torch.nn import Linear

class TestLinearLayer(unittest.TestCase):
    
    # 1º Ciclo: Teste de falha para entrada com dimensão incorreta
    def test_forward_shape_mismatch(self):
        model = Linear(10, 5)
        input_tensor = torch.randn(2, 15)  # Dimensão errada
        with self.assertRaises(RuntimeError):
            model(input_tensor)
    
    # 2º Ciclo: Teste de sucesso para entrada com dimensão correta
    def test_forward_success(self):
        model = Linear(10, 5)
        input_tensor = torch.randn(2, 10)  # Dimensão correta
        output = model(input_tensor)
        self.assertEqual(output.shape, (2, 5))
    
    # 3º Ciclo: Teste para verificar inicialização dos pesos
    def test_weight_initialization(self):
        model = Linear(10, 5)
        self.assertEqual(model.weight.shape, (5, 10))

if __name__ == "__main__":
    unittest.main()
