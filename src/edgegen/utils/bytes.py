from dataclasses import dataclass

@dataclass()
class Bytes():
    size: int

    def __add__(self, other:'Bytes') -> 'Bytes':
        return Bytes(self.size + other.size)

    def __sub__(self, other:'Bytes') -> 'Bytes':
        return Bytes(self.size - other.size)
    
    def __mul__(self, other:int) -> 'Bytes':
        return Bytes(self.size * other)
    
    def __max__(self, other:'Bytes') -> 'Bytes':
        return Bytes(max(self.size, other.size))
    
    def __eq__(self, other: 'Bytes') -> bool:
        return self.size == other.size

    def __le__(self, other: 'Bytes') -> bool:
        return self.size <= other.size

    def __lt__(self, other: 'Bytes') -> bool:
        return self.size < other.size

    def __ge__(self, other: 'Bytes') -> bool:
        return self.size >= other.size
    
    def __gt__(self, other: 'Bytes') -> bool:
        return self.size > other.size

    def to_KB(self):
        return self.size / 1024

    def to_MB(self):
        return self.size / (1024 ** 2)
    
    def to_GB(self):
        return self.size / (1024 ** 3)
    
    @classmethod
    def from_KB(self, size):
        return Bytes(size * 1024)
    
    @classmethod
    def from_MB(self, size):
        return Bytes(size * (1024 ** 2))
    
    @classmethod
    def from_GB(self, size):
        return Bytes(size * (1024 ** 3))

def __str__(self):
    # print the size in bytes, KB, MB, and GB in pretty format on multiple lines
    return f"Size in bytes: {self.size}\nSize in KB: {self.to_KB()}\nSize in MB: {self.to_MB()}\nSize in GB: {self.to_GB()}"
