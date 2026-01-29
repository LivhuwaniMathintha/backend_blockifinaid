// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

contract StudentFunding {
    // --- Structs ---
    struct Institution {
        uint256 id;
        string name;
        uint256 yearOpened;
        string summary;
        bool isActive;
    }

    struct Student {
        uint256 id;
        uint256 studentNumber;
        uint256 institutionId;
        uint256 courseId;
        string courseName;
        uint256 year;
        bool isActive;
        bool isFunded;
        uint256 funderId;
        uint256 funderContractId;
        uint256 studentBankAccountDetailsId;
    }

    struct StudentBankAccountDetails {
        uint256 id;
        uint256 bankAccountNumber;
        string bankName;
        string bankBranchCode;
        bool isConfirmed;
        uint256 dataConfirmedById;
    }

    struct FundingContract {
        uint256 id;
        uint256 funderId;
        uint256 studentId;
        uint256 funderContractConditionsId;
        uint256 signedOn;
        bool isActive;
        uint256 foodBalance;
        uint256 tuitionBalance;
        uint256 laptopBalance;
    }

    struct FunderContractConditions {
        uint256 id;
        uint256 studentId;
        uint256 funderId;
        bool isFullCoverage;
        uint256 minimumYearsFunded;
        uint256 maxYearsFunded;
        uint256 firstSemesterAverage;
        uint256 lastSemesterAverage;
        uint256 fullYearAverage;
        uint256 averageYear;
        uint256 totalAmount;
        uint256 foodAmount;
        uint256 tuitionAmount;
        uint256 laptopBookAmount;
        uint256 accommodationAmount;
        uint256 accommodationDirectPay;
        uint256 studentAccommodationDetailsId;
        uint256 monthlyFoodAmount;
        uint256 monthlyAccommodationAmount;
        uint256 foodPaymentStatus;
        uint256 paymentDate;
        uint256 studentBankAccountDetailsId;
    }

    struct PaymentStatusCode {
        uint256 id;
        string status;
    }

    struct StudentAccommodationDetails {
        uint256 id;
        string homeAddress;
        uint256 postalCode;
        string suburb;
        string ownerFullname;
        string ownerEmail;
        string ownerNumber;
        uint256 studentId;
    }

    struct StudentPayments {
        uint256 studentId;
        uint256 studentPaymentId;
        string summary;
    }

    struct PaymentType {
        uint256 id;
        string title;
    }

    struct Payment {
        uint256 id;
        uint256 fundingContractId;
        uint256 paymentTypeId;
        bool paidExternally;
    }

    struct AuditTrail {
        uint256 id;
        string tableName;
        string operation;
        string preImage;
        string postImage;
        string ipAddress;
        string machineType;
        string message;
    }

    // --- Mappings ---
    mapping(uint256 => Institution) public institutions;
    mapping(uint256 => Student) public students;
    mapping(uint256 => StudentBankAccountDetails) public studentBankAccounts;
    mapping(uint256 => FundingContract) public fundingContracts;
    mapping(uint256 => FunderContractConditions) public funderContractConditions;
    mapping(uint256 => PaymentStatusCode) public paymentStatusCodes;
    mapping(uint256 => StudentAccommodationDetails) public studentAccommodationDetails;
    mapping(bytes32 => StudentPayments) public studentPayments; // composite key: keccak256(studentId, studentPaymentId)
    mapping(uint256 => PaymentType) public paymentTypes;
    mapping(uint256 => Payment) public payments;
    mapping(uint256 => AuditTrail) public auditTrails;
    uint256 public auditTrailCount;

    // --- Auto-incrementing IDs ---
    uint256 public lastInstitutionId;
    uint256 public lastStudentId;
    uint256 public lastStudentBankAccountId;
    uint256 public lastFundingContractId;
    uint256 public lastFunderContractConditionId;
    uint256 public lastPaymentStatusCodeId;
    uint256 public lastStudentAccommodationDetailsId;
    uint256 public lastPaymentTypeId;
    uint256 public lastPaymentId;

    // --- Helper: Composite Key ---
    function getStudentPaymentsKey(uint256 studentId, uint256 studentPaymentId) internal pure returns (bytes32) {
        return keccak256(abi.encodePacked(studentId, studentPaymentId));
    }

    // --- Storage for IDs for retrieval ---
    uint256[] private institutionIds;
    uint256[] private studentIds;
    uint256[] private studentBankAccountIds;
    uint256[] private fundingContractIds;
    uint256[] private funderContractConditionIds;
    uint256[] private paymentStatusCodeIds;
    uint256[] private studentAccommodationDetailsIds;
    bytes32[] private studentPaymentsKeys;
    uint256[] private paymentTypeIds;
    uint256[] private paymentIds;

    // --- Add Methods (auto-increment IDs) ---

    function addInstitution(
        string memory name,
        uint256 yearOpened,
        string memory summary,
        bool isActive,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public returns (uint256) {
        lastInstitutionId++;
        uint256 id = lastInstitutionId;
        institutions[id] = Institution(id, name, yearOpened, summary, isActive);
        institutionIds.push(id);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "Institution",
            "Insert",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
        return id;
    }

    function addStudent(
        uint256 studentNumber,
        uint256 institutionId,
        uint256 courseId,
        string memory courseName,
        uint256 year,
        bool isActive,
        bool isFunded,
        uint256 funderId,
        uint256 funderContractId,
        uint256 studentBankAccountDetailsId,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public returns (uint256) {
        lastStudentId++;
        uint256 id = lastStudentId;
        students[id] = Student(
            id,
            studentNumber,
            institutionId,
            courseId,
            courseName,
            year,
            isActive,
            isFunded,
            funderId,
            funderContractId,
            studentBankAccountDetailsId
        );
        studentIds.push(id);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "Student",
            "Insert",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
        return id;
    }

    function addStudentBankAccount(
        uint256 bankAccountNumber,
        string memory bankName,
        string memory bankBranchCode,
        bool isConfirmed,
        uint256 dataConfirmedById,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public returns (uint256) {
        lastStudentBankAccountId++;
        uint256 id = lastStudentBankAccountId;
        studentBankAccounts[id] = StudentBankAccountDetails(
            id,
            bankAccountNumber,
            bankName,
            bankBranchCode,
            isConfirmed,
            dataConfirmedById
        );
        studentBankAccountIds.push(id);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "StudentBankAccountDetails",
            "Insert",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
        return id;
    }

    function addFundingContract(
        uint256 funderId,
        uint256 studentId,
        uint256 funderContractConditionsId,
        uint256 signedOn,
        bool isActive,
        uint256 foodBalance,
        uint256 tuitionBalance,
        uint256 laptopBalance,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public returns (uint256) {
        lastFundingContractId++;
        uint256 id = lastFundingContractId;
        fundingContracts[id] = FundingContract(
            id,
            funderId,
            studentId,
            funderContractConditionsId,
            signedOn,
            isActive,
            foodBalance,
            tuitionBalance,
            laptopBalance
        );
        fundingContractIds.push(id);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "FundingContract",
            "Insert",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
        return id;
    }

    function addFunderContractConditions(
        uint256 studentId,
        uint256 funderId,
        bool isFullCoverage,
        uint256 minimumYearsFunded,
        uint256 maxYearsFunded,
        uint256 firstSemesterAverage,
        uint256 lastSemesterAverage,
        uint256 fullYearAverage,
        uint256 averageYear,
        uint256 totalAmount,
        uint256 foodAmount,
        uint256 tuitionAmount,
        uint256 laptopBookAmount,
        uint256 accommodationAmount,
        uint256 accommodationDirectPay,
        uint256 studentAccommodationDetailsId,
        uint256 monthlyFoodAmount,
        uint256 monthlyAccommodationAmount,
        uint256 foodPaymentStatus,
        uint256 paymentDate,
        uint256 studentBankAccountDetailsId,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public returns (uint256) {
        lastFunderContractConditionId++;
        uint256 id = lastFunderContractConditionId;
        funderContractConditions[id] = FunderContractConditions(
            id,
            studentId,
            funderId,
            isFullCoverage,
            minimumYearsFunded,
            maxYearsFunded,
            firstSemesterAverage,
            lastSemesterAverage,
            fullYearAverage,
            averageYear,
            totalAmount,
            foodAmount,
            tuitionAmount,
            laptopBookAmount,
            accommodationAmount,
            accommodationDirectPay,
            studentAccommodationDetailsId,
            monthlyFoodAmount,
            monthlyAccommodationAmount,
            foodPaymentStatus,
            paymentDate,
            studentBankAccountDetailsId
        );
        funderContractConditionIds.push(id);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "FunderContractConditions",
            "Insert",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
        return id;
    }

    function addPaymentStatusCode(
        string memory status,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public returns (uint256) {
        lastPaymentStatusCodeId++;
        uint256 id = lastPaymentStatusCodeId;
        paymentStatusCodes[id] = PaymentStatusCode(id, status);
        paymentStatusCodeIds.push(id);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "PaymentStatusCode",
            "Insert",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
        return id;
    }

    function addStudentAccommodationDetails(
        string memory homeAddress,
        uint256 postalCode,
        string memory suburb,
        string memory ownerFullname,
        string memory ownerEmail,
        string memory ownerNumber,
        uint256 studentId,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public returns (uint256) {
        lastStudentAccommodationDetailsId++;
        uint256 id = lastStudentAccommodationDetailsId;
        studentAccommodationDetails[id] = StudentAccommodationDetails(
            id,
            homeAddress,
            postalCode,
            suburb,
            ownerFullname,
            ownerEmail,
            ownerNumber,
            studentId
        );
        studentAccommodationDetailsIds.push(id);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "StudentAccommodationDetails",
            "Insert",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
        return id;
    }

    function addPaymentType(
        string memory title,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public returns (uint256) {
        lastPaymentTypeId++;
        uint256 id = lastPaymentTypeId;
        paymentTypes[id] = PaymentType(id, title);
        paymentTypeIds.push(id);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "PaymentType",
            "Insert",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
        return id;
    }

    function addPayment(
        uint256 fundingContractId,
        uint256 paymentTypeId,
        bool paidExternally,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public returns (uint256) {
        lastPaymentId++;
        uint256 id = lastPaymentId;
        payments[id] = Payment(id, fundingContractId, paymentTypeId, paidExternally);
        paymentIds.push(id);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "Payment",
            "Insert",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
        return id;
    }

    // --- Update Methods (require id as input) ---

    function updateInstitution(
        uint256 id,
        string memory name,
        uint256 yearOpened,
        string memory summary,
        bool isActive,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public {
        require(institutions[id].id != 0, "Institution not found");
        institutions[id] = Institution(id, name, yearOpened, summary, isActive);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "Institution",
            "Update",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
    }

    function updateStudent(
        uint256 id,
        uint256 studentNumber,
        uint256 institutionId,
        uint256 courseId,
        string memory courseName,
        uint256 year,
        bool isActive,
        bool isFunded,
        uint256 funderId,
        uint256 funderContractId,
        uint256 studentBankAccountDetailsId,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public {
        require(students[id].id != 0, "Student not found");
        students[id] = Student(
            id,
            studentNumber,
            institutionId,
            courseId,
            courseName,
            year,
            isActive,
            isFunded,
            funderId,
            funderContractId,
            studentBankAccountDetailsId
        );
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "Student",
            "Update",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
    }

    function updateStudentBankAccount(
        uint256 id,
        uint256 bankAccountNumber,
        string memory bankName,
        string memory bankBranchCode,
        bool isConfirmed,
        uint256 dataConfirmedById,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public {
        require(studentBankAccounts[id].id != 0, "Bank account not found");
        studentBankAccounts[id] = StudentBankAccountDetails(
            id,
            bankAccountNumber,
            bankName,
            bankBranchCode,
            isConfirmed,
            dataConfirmedById
        );
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "StudentBankAccountDetails",
            "Update",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
    }

    function updateFundingContract(
        uint256 id,
        uint256 funderId,
        uint256 studentId,
        uint256 funderContractConditionsId,
        uint256 signedOn,
        bool isActive,
        uint256 foodBalance,
        uint256 tuitionBalance,
        uint256 laptopBalance,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public {
        require(fundingContracts[id].id != 0, "FundingContract not found");
        fundingContracts[id] = FundingContract(
            id,
            funderId,
            studentId,
            funderContractConditionsId,
            signedOn,
            isActive,
            foodBalance,
            tuitionBalance,
            laptopBalance
        );
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "FundingContract",
            "Update",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
    }

    function updateFunderContractConditions(
        uint256 id,
        uint256 studentId,
        uint256 funderId,
        bool isFullCoverage,
        uint256 minimumYearsFunded,
        uint256 maxYearsFunded,
        uint256 firstSemesterAverage,
        uint256 lastSemesterAverage,
        uint256 fullYearAverage,
        uint256 averageYear,
        uint256 totalAmount,
        uint256 foodAmount,
        uint256 tuitionAmount,
        uint256 laptopBookAmount,
        uint256 accommodationAmount,
        uint256 accommodationDirectPay,
        uint256 studentAccommodationDetailsId,
        uint256 monthlyFoodAmount,
        uint256 monthlyAccommodationAmount,
        uint256 foodPaymentStatus,
        uint256 paymentDate,
        uint256 studentBankAccountDetailsId,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public {
        require(funderContractConditions[id].id != 0, "FunderContractConditions not found");
        funderContractConditions[id] = FunderContractConditions(
            id,
            studentId,
            funderId,
            isFullCoverage,
            minimumYearsFunded,
            maxYearsFunded,
            firstSemesterAverage,
            lastSemesterAverage,
            fullYearAverage,
            averageYear,
            totalAmount,
            foodAmount,
            tuitionAmount,
            laptopBookAmount,
            accommodationAmount,
            accommodationDirectPay,
            studentAccommodationDetailsId,
            monthlyFoodAmount,
            monthlyAccommodationAmount,
            foodPaymentStatus,
            paymentDate,
            studentBankAccountDetailsId
        );
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "FunderContractConditions",
            "Update",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
    }

    function updatePaymentStatusCode(
        uint256 id,
        string memory status,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public {
        require(paymentStatusCodes[id].id != 0, "PaymentStatusCode not found");
        paymentStatusCodes[id] = PaymentStatusCode(id, status);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "PaymentStatusCode",
            "Update",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
    }

    function updateStudentAccommodationDetails(
        uint256 id,
        string memory homeAddress,
        uint256 postalCode,
        string memory suburb,
        string memory ownerFullname,
        string memory ownerEmail,
        string memory ownerNumber,
        uint256 studentId,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public {
        require(studentAccommodationDetails[id].id != 0, "StudentAccommodationDetails not found");
        studentAccommodationDetails[id] = StudentAccommodationDetails(
            id,
            homeAddress,
            postalCode,
            suburb,
            ownerFullname,
            ownerEmail,
            ownerNumber,
            studentId
        );
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "StudentAccommodationDetails",
            "Update",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
    }

    function updatePaymentType(
        uint256 id,
        string memory title,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public {
        require(paymentTypes[id].id != 0, "PaymentType not found");
        paymentTypes[id] = PaymentType(id, title);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "PaymentType",
            "Update",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
    }

    function updatePayment(
        uint256 id,
        uint256 fundingContractId,
        uint256 paymentTypeId,
        bool paidExternally,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public {
        require(payments[id].id != 0, "Payment not found");
        payments[id] = Payment(id, fundingContractId, paymentTypeId, paidExternally);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "Payment",
            "Update",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
    }

    // --- StudentPayments (composite key) ---
    function addOrUpdateStudentPayments(
        uint256 studentId,
        string memory summary,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message,
        bool isUpdate
    ) public returns (uint256) {
        // Auto-increment studentPaymentId per student
        uint256 studentPaymentId = 1;
        if (isUpdate) {
            // For update, client must provide the correct studentPaymentId
            revert("Use updateStudentPayments for updates with studentPaymentId");
        } else {
            // Find the next available studentPaymentId for this student
            while (true) {
                bytes32 key = getStudentPaymentsKey(studentId, studentPaymentId);
                if (studentPayments[key].studentId == 0) {
                    // Not used
                    studentPayments[key] = StudentPayments(studentId, studentPaymentId, summary);
                    studentPaymentsKeys.push(key);
                    auditTrails[auditTrailCount++] = AuditTrail(
                        auditTrailCount,
                        "StudentPayments",
                        "Insert",
                        preImage,
                        postImage,
                        ipAddress,
                        machineType,
                        message
                    );
                    return studentPaymentId;
                }
                studentPaymentId++;
            }
        }
    }

    function updateStudentPayments(
        uint256 studentId,
        uint256 studentPaymentId,
        string memory summary,
        string memory preImage,
        string memory postImage,
        string memory ipAddress,
        string memory machineType,
        string memory message
    ) public {
        bytes32 key = getStudentPaymentsKey(studentId, studentPaymentId);
        require(studentPayments[key].studentId != 0, "StudentPayments not found");
        studentPayments[key] = StudentPayments(studentId, studentPaymentId, summary);
        auditTrails[auditTrailCount++] = AuditTrail(
            auditTrailCount,
            "StudentPayments",
            "Update",
            preImage,
            postImage,
            ipAddress,
            machineType,
            message
        );
    }

    // --- Retrieve by ID ---
    function getInstitutionById(uint256 id) public view returns (Institution memory) {
        return institutions[id];
    }
    function getStudentById(uint256 id) public view returns (Student memory) {
        return students[id];
    }
    function getStudentBankAccountById(uint256 id) public view returns (StudentBankAccountDetails memory) {
        return studentBankAccounts[id];
    }
    function getFundingContractById(uint256 id) public view returns (FundingContract memory) {
        return fundingContracts[id];
    }
    function getFunderContractConditionsById(uint256 id) public view returns (FunderContractConditions memory) {
        return funderContractConditions[id];
    }
    function getPaymentStatusCodeById(uint256 id) public view returns (PaymentStatusCode memory) {
        return paymentStatusCodes[id];
    }
    function getStudentAccommodationDetailsById(uint256 id) public view returns (StudentAccommodationDetails memory) {
        return studentAccommodationDetails[id];
    }
    function getStudentPaymentsById(uint256 studentId, uint256 studentPaymentId) public view returns (StudentPayments memory) {
        return studentPayments[getStudentPaymentsKey(studentId, studentPaymentId)];
    }
    function getPaymentTypeById(uint256 id) public view returns (PaymentType memory) {
        return paymentTypes[id];
    }
    function getPaymentById(uint256 id) public view returns (Payment memory) {
        return payments[id];
    }
    function getAuditTrailById(uint256 id) public view returns (AuditTrail memory) {
        return auditTrails[id];
    }

    // --- Retrieve All ---
    function getAllInstitutions() public view returns (Institution[] memory) {
        Institution[] memory result = new Institution[](institutionIds.length);
        for (uint256 i = 0; i < institutionIds.length; i++) {
            result[i] = institutions[institutionIds[i]];
        }
        return result;
    }
    function getAllStudents() public view returns (Student[] memory) {
        Student[] memory result = new Student[](studentIds.length);
        for (uint256 i = 0; i < studentIds.length; i++) {
            result[i] = students[studentIds[i]];
        }
        return result;
    }
    function getAllStudentBankAccounts() public view returns (StudentBankAccountDetails[] memory) {
        StudentBankAccountDetails[] memory result = new StudentBankAccountDetails[](studentBankAccountIds.length);
        for (uint256 i = 0; i < studentBankAccountIds.length; i++) {
            result[i] = studentBankAccounts[studentBankAccountIds[i]];
        }
        return result;
    }
    function getAllFundingContracts() public view returns (FundingContract[] memory) {
        FundingContract[] memory result = new FundingContract[](fundingContractIds.length);
        for (uint256 i = 0; i < fundingContractIds.length; i++) {
            result[i] = fundingContracts[fundingContractIds[i]];
        }
        return result;
    }
    function getAllFunderContractConditions() public view returns (FunderContractConditions[] memory) {
        FunderContractConditions[] memory result = new FunderContractConditions[](funderContractConditionIds.length);
        for (uint256 i = 0; i < funderContractConditionIds.length; i++) {
            result[i] = funderContractConditions[funderContractConditionIds[i]];
        }
        return result;
    }
    function getAllPaymentStatusCodes() public view returns (PaymentStatusCode[] memory) {
        PaymentStatusCode[] memory result = new PaymentStatusCode[](paymentStatusCodeIds.length);
        for (uint256 i = 0; i < paymentStatusCodeIds.length; i++) {
            result[i] = paymentStatusCodes[paymentStatusCodeIds[i]];
        }
        return result;
    }
    function getAllStudentAccommodationDetails() public view returns (StudentAccommodationDetails[] memory) {
        StudentAccommodationDetails[] memory result = new StudentAccommodationDetails[](studentAccommodationDetailsIds.length);
        for (uint256 i = 0; i < studentAccommodationDetailsIds.length; i++) {
            result[i] = studentAccommodationDetails[studentAccommodationDetailsIds[i]];
        }
        return result;
    }
    function getAllStudentPayments() public view returns (StudentPayments[] memory) {
        StudentPayments[] memory result = new StudentPayments[](studentPaymentsKeys.length);
        for (uint256 i = 0; i < studentPaymentsKeys.length; i++) {
            result[i] = studentPayments[studentPaymentsKeys[i]];
        }
        return result;
    }
    function getAllPaymentTypes() public view returns (PaymentType[] memory) {
        PaymentType[] memory result = new PaymentType[](paymentTypeIds.length);
        for (uint256 i = 0; i < paymentTypeIds.length; i++) {
            result[i] = paymentTypes[paymentTypeIds[i]];
        }
        return result;
    }
    function getAllPayments() public view returns (Payment[] memory) {
        Payment[] memory result = new Payment[](paymentIds.length);
        for (uint256 i = 0; i < paymentIds.length; i++) {
            result[i] = payments[paymentIds[i]];
        }
        return result;
    }
    function getAllAuditTrails() public view returns (AuditTrail[] memory) {
        AuditTrail[] memory result = new AuditTrail[](auditTrailCount);
        for (uint256 i = 0; i < auditTrailCount; i++) {
            result[i] = auditTrails[i];
        }
        return result;
    }
}