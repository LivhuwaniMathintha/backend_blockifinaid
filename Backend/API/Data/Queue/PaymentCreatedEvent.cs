namespace API.Data.Queue;

public record PaymentCreatedEvent(string? PaymentId, string StuduentEmail,
    string StudentName,
    string UserId,
    string StudnetBankAccountNumber,
    decimal Amount,
    string PaymentType,
    string StudentBankName,
    string StudentBankBranchCode,
    string OrgBankAccountNumber,
    string OrgBankName,
    string OrgBankBranchCode,
    DateTime CreatedAt,
    bool IsPaid = false)
{ }


  